"""
created matt_dumont 
on: 23/05/24
"""
import unittest
import datetime
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from komanawa.kendall_stats.mann_kendall import _mann_kendall_from_sarray, _make_s_array, _mann_kendall_old, \
    _seasonal_mann_kendall_from_sarray, _old_smk, MannKendall, SeasonalKendall, \
    _calc_seasonal_senslope
from komanawa.kendall_stats.example_data import make_increasing_decreasing_data, make_seasonal_data
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TestMannKendall(unittest.TestCase):
    def _quick_test_s(self):
        np.random.seed(54)
        x = np.random.rand(100)
        s_new = _make_s_array(x).sum()
        s = 0
        n = len(x)
        for k in range(n - 1):
            for j in range(k + 1, n):
                s += np.sign(x[j] - x[k])
        self.assertEqual(s_new, s)

    def test_new_old_mann_kendall(self):
        np.random.seed(54)
        x = np.random.rand(100)
        trend, h, p, z, s, var_s = _mann_kendall_from_sarray(x)
        trend_old, h_old, p_old, z_old, s_old, var_s_old = _mann_kendall_old(x)
        self.assertEqual(trend, trend_old)
        self.assertEqual(h, h_old)
        self.assertEqual(p, p_old)
        self.assertEqual(z, z_old)
        self.assertEqual(s, s_old)
        self.assertEqual(var_s, var_s_old)

    def test_part_mann_kendall(self):
        np.random.seed(54)
        x = np.random.rand(500)
        s_array_full = _make_s_array(x)
        for i in np.arange(8, 400):
            old = _mann_kendall_from_sarray(x[i:])
            new = _mann_kendall_from_sarray(x[i:], sarray=s_array_full[i:, i:])
            self.assertEqual(old, new)

    def test_seasonal_kendall_sarray(self):
        np.random.seed(54)
        x = np.random.rand(500)
        seasons = np.repeat([np.arange(4)], 125, axis=0).flatten()
        new = _seasonal_mann_kendall_from_sarray(x, seasons)
        old = _old_smk(pd.DataFrame(dict(x=x, seasons=seasons)), 'x', 'seasons')
        self.assertEqual(new, old)

    def test_seasonal_data(self):
        save_path = Path(__file__).parent.joinpath('test_data', 'test_seasonal_data.hdf')
        write_test_data = False
        slope = 0.1
        noise = 10
        for sort, na_data in itertools.product([True, False], [True, False]):
            test_dataframe = make_seasonal_data(slope, noise, sort, na_data)
            test_name = f'slope_{slope}_noise_{noise}_unsort_{sort}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                 '_')
            if write_test_data:
                test_dataframe.to_hdf(save_path, test_name)
            else:
                expect = pd.read_hdf(save_path, test_name)
                self.assertIsInstance(expect, pd.DataFrame)
                pd.testing.assert_frame_equal(test_dataframe, expect, check_names=False, obj=test_name)
        for na_data in [True, False]:
            sort_name = f'slope_{slope}_noise_{noise}_unsort_{False}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                  '_')
            sort_data = pd.read_hdf(save_path, sort_name)
            self.assertIsInstance(sort_data, pd.DataFrame)
            sort_data.name = sort_name
            unsort_name = f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                   '_')
            unsort_data = pd.read_hdf(save_path, unsort_name)
            self.assertIsInstance(unsort_data, pd.DataFrame)
            unsort_data.name = unsort_name
            unsort_data = unsort_data.sort_index()
            pd.testing.assert_frame_equal(sort_data, unsort_data, check_names=False, obj=sort_name)
            self.assertTrue(np.allclose(sort_data.values, unsort_data.values, equal_nan=True))

    def test_mann_kendall(self, show=False):
        test_data_path = Path(__file__).parent.joinpath('test_data', 'test_mk.hdf')
        make_test_data = False
        slopes = [0.1, -0.1, 0]
        noises = [5, 10, 50]
        unsorts = [True, False]
        na_datas = [True, False]
        figs = []

        for slope, noise, unsort, na_data in itertools.product(slopes, noises, unsorts, na_datas):

            x, y = make_increasing_decreasing_data(slope=slope, noise=noise)
            if na_data:
                np.random.seed(868)
                na_idxs = np.random.randint(0, len(y), 10)
                y[na_idxs] = np.nan

            # test passing numpy array
            mk_array = MannKendall(data=y, alpha=0.05, data_col=None, rm_na=True)

            # test passing Series
            test_data = pd.Series(y, index=x)
            if unsort:
                x_use = deepcopy(x)
                np.random.shuffle(x_use)
                test_data = test_data.loc[x_use]
            mk_series = MannKendall(data=test_data, alpha=0.05, data_col=None, rm_na=True)

            # test passing data col (with other noisy cols)
            test_dataframe = pd.DataFrame(index=x, data=y, columns=['y'])
            for col in ['lkj', 'lskdfj', 'laskdfj']:
                test_dataframe[col] = np.random.choice([1, 34.2, np.nan])
            if unsort:
                x_use = deepcopy(x)
                np.random.shuffle(x_use)
                test_dataframe = test_dataframe.loc[x_use]
            mk_df = MannKendall(data=test_dataframe, alpha=0.05, data_col='y', rm_na=True)

            # test results
            self.assertTrue(mk_array.trend == mk_series.trend == mk_df.trend)
            self.assertTrue(mk_array.h == mk_series.h == mk_df.h)
            self.assertTrue(np.allclose(mk_array.p, mk_series.p))
            self.assertTrue(np.allclose(mk_array.p, mk_df.p))

            self.assertTrue(np.allclose(mk_array.z, mk_series.z))
            self.assertTrue(np.allclose(mk_array.z, mk_df.z))

            self.assertTrue(np.allclose(mk_array.s, mk_df.s))
            self.assertTrue(np.allclose(mk_array.s, mk_df.s))

            self.assertTrue(np.allclose(mk_array.var_s, mk_series.var_s))
            self.assertTrue(np.allclose(mk_array.var_s, mk_df.var_s))

            # senslopes
            array_ss_data = np.array(mk_array.calc_senslope())
            series_ss_data = np.array(mk_series.calc_senslope())
            df_ss_data = np.array(mk_df.calc_senslope())
            self.assertTrue(np.allclose(array_ss_data, series_ss_data))
            self.assertTrue(np.allclose(array_ss_data, df_ss_data))

            got_data = pd.Series(dict(trend=mk_array.trend, h=mk_array.h, p=mk_array.p, z=mk_array.z, s=mk_array.s,
                                      var_s=mk_array.var_s, senslope=array_ss_data[0],
                                      sen_intercept=array_ss_data[1],
                                      lo_slope=array_ss_data[2],
                                      up_slope=array_ss_data[3], ))
            got_data = got_data.astype(float)
            test_name = f'slope_{slope}_noise_{noise}_unsort_{unsort}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                   '_')
            # test plot data
            fig, ax, (handles, labels) = mk_array.plot_data()
            figs.append(fig)
            ax.set_title(test_name)

            if not make_test_data:
                test_data = pd.read_hdf(test_data_path, test_name)
                self.assertIsInstance(test_data, pd.Series)
                pd.testing.assert_series_equal(got_data, test_data, check_names=False)
            else:
                got_data.to_hdf(test_data_path, test_name)

            # test that sort vs unsort doesn't change results
        for slope, noise, na_data in itertools.product(slopes, noises, na_datas):
            sort_name = f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                 '_')
            sort_data = pd.read_hdf(test_data_path, sort_name)
            unsort_name = f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                   '_')
            unsort_data = pd.read_hdf(test_data_path, unsort_name)
            self.assertIsInstance(sort_data, pd.Series)
            self.assertIsInstance(unsort_data, pd.Series)
            pd.testing.assert_series_equal(sort_data, unsort_data, check_names=False)

        # test mann kendall plot (with datetime)
        x, y = make_increasing_decreasing_data(slope=0.1, noise=10)
        data = pd.Series(y, index=datetime.datetime.today() + pd.to_timedelta(x * 7, unit='D'))
        t = MannKendall(data=data, alpha=0.05, data_col=None, rm_na=True)
        t.plot_data()
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_seasonal_senslope(self):
        seasonal_data_base = make_seasonal_data(slope=0.1, noise=10, unsort=False, na_data=False)
        seasonal_data1 = deepcopy(seasonal_data_base)
        seasonal_data2 = deepcopy(seasonal_data_base)

        t1 = _calc_seasonal_senslope(y=seasonal_data1['y'], season=seasonal_data1['seasons'], x=seasonal_data1.index,
                                     alpha=0.05)
        t2 = _calc_seasonal_senslope(y=seasonal_data2['y'], season=seasonal_data2['seasons'], x=seasonal_data2.index,
                                     alpha=0.05)
        self.assertTrue(np.allclose(t1, t2))

        # test sort vs unsort
        seasonal_data1 = deepcopy(seasonal_data_base)
        seasonal_data2 = make_seasonal_data(slope=0.1, noise=10, unsort=True, na_data=False)
        seasonal_data2 = seasonal_data2.sort_index()
        self.assertTrue(np.allclose(seasonal_data1.values, seasonal_data2.values, equal_nan=True))
        t1 = _calc_seasonal_senslope(y=seasonal_data1['y'], season=seasonal_data1['seasons'], x=seasonal_data1.index,
                                     alpha=0.05)
        t2 = _calc_seasonal_senslope(y=seasonal_data2['y'], season=seasonal_data2['seasons'], x=seasonal_data2.index,
                                     alpha=0.05)
        self.assertTrue(np.allclose(t1, t2))

        # test sort vs unsort with nan
        seasonal_data1 = make_seasonal_data(slope=0.1, noise=10, unsort=False, na_data=True)
        seasonal_data2 = make_seasonal_data(slope=0.1, noise=10, unsort=True, na_data=True)
        seasonal_data2 = seasonal_data2.sort_index()
        self.assertTrue(np.allclose(seasonal_data1.values, seasonal_data2.values, equal_nan=True))
        seasonal_data1 = seasonal_data1.dropna(subset=['y', 'seasons'])
        seasonal_data2 = seasonal_data2.dropna(subset=['y', 'seasons'])
        self.assertTrue(np.allclose(seasonal_data1.values, seasonal_data2.values, equal_nan=True))
        t1 = _calc_seasonal_senslope(y=seasonal_data1['y'], season=seasonal_data1['seasons'], x=seasonal_data1.index,
                                     alpha=0.05)
        t2 = _calc_seasonal_senslope(y=seasonal_data2['y'], season=seasonal_data2['seasons'], x=seasonal_data2.index,
                                     alpha=0.05)
        self.assertTrue(np.allclose(t1, t2))

    def test_seasonal_mann_kendall(self, show=False):
        test_data_path = Path(__file__).parent.joinpath('test_data', 'test_smk.hdf')
        make_test_data = False
        slopes = [0.1, -0.1, 0]
        noises = [5, 10, 50]
        unsorts = [True, False]
        na_datas = [True, False]
        figs = []
        for slope, noise, unsort, na_data in itertools.product(slopes, noises, unsorts, na_datas):
            test_dataframe = make_seasonal_data(slope, noise, unsort, na_data)

            mk_df = SeasonalKendall(df=test_dataframe, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                    freq_limit=0.05)

            # test results
            df_ss_data = np.array(mk_df.calc_senslope())

            got_data = pd.Series(dict(trend=mk_df.trend, h=mk_df.h, p=mk_df.p, z=mk_df.z, s=mk_df.s,
                                      var_s=mk_df.var_s, senslope=df_ss_data[0],
                                      sen_intercept=df_ss_data[1],
                                      lo_slope=df_ss_data[2],
                                      up_slope=df_ss_data[3], ))
            got_data = got_data.astype(float)
            test_name = f'slope_{slope}_noise_{noise}_unsort_{unsort}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                   '_')
            # test plot data
            fig, ax, (handles, labels) = mk_df.plot_data()
            figs.append(fig)
            ax.set_title(test_name)

            if not make_test_data:
                test_data = pd.read_hdf(test_data_path, test_name)
                self.assertIsInstance(test_data, pd.Series)
                pd.testing.assert_series_equal(got_data, test_data, check_names=False, obj=test_name)
            else:
                got_data.to_hdf(test_data_path, test_name, complevel=9, complib='blosc:lz4')

            # test that sort vs unsort doesn't change results
        for slope, noise, na_data in itertools.product(slopes, noises, na_datas):
            sort_name = f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                 '_')
            sort_data = pd.read_hdf(test_data_path, sort_name)
            sort_data.name = sort_name
            unsort_name = f'slope_{slope}_noise_{noise}_unsort_{False}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                    '_')
            unsort_data = pd.read_hdf(test_data_path, unsort_name)
            unsort_data.name = unsort_name
            self.assertIsInstance(sort_data, pd.Series)
            self.assertIsInstance(unsort_data, pd.Series)
            pd.testing.assert_series_equal(sort_data, unsort_data, check_names=False,
                                           obj=f'{sort_name} & {unsort_name}')

        # test seasonal mann kendall plot (with datetime)
        data = make_seasonal_data(slope=0.1, noise=10, unsort=False, na_data=False)
        data.index = datetime.datetime.today() + pd.to_timedelta(data.index * 7, unit='D')
        mk_df = SeasonalKendall(df=data, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                freq_limit=0.05)
        mk_df.plot_data()
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
