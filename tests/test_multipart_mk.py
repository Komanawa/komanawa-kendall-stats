"""
created matt_dumont 
on: 23/05/24
"""
import unittest
import datetime
import itertools
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from komanawa.kendall_stats.multi_part_kendall import MultiPartKendall, SeasonalMultiPartKendall, _generate_startpoints
from komanawa.kendall_stats.mann_kendall import _mann_kendall_from_sarray, _make_s_array, _mann_kendall_old, \
    _seasonal_mann_kendall_from_sarray, _old_smk, MannKendall, SeasonalKendall, \
    _calc_seasonal_senslope, get_colors
from komanawa.kendall_stats.example_data import make_increasing_decreasing_data, make_seasonal_data, \
    make_multipart_sharp_change_data, multipart_sharp_slopes, multipart_sharp_noises, \
    make_multipart_parabolic_data, multipart_parabolic_slopes, multipart_parabolic_noises, \
    make_seasonal_multipart_parabolic, make_seasonal_multipart_sharp_change
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TestMultipartMK(unittest.TestCase):

    def test_plot_multipart_data_sharp(self, show=False):
        # sharp change
        f = make_multipart_sharp_change_data
        colors = get_colors(multipart_sharp_noises)
        figs = []
        for slope in multipart_sharp_slopes:
            for noise, c in zip(multipart_sharp_noises, colors):
                fig, ax = plt.subplots()
                figs.append(fig)
                x, y = f(slope, noise, unsort=False, na_data=False)
                ax.scatter(x, y, label=f'noise:{noise}', c=c)
                ax.legend()
                ax.set_title(f'slope:{slope}, f:{f.__name__}')
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_plot_multipart_data_para(self, show=False):
        # parabolic
        f = make_multipart_parabolic_data
        colors = get_colors(multipart_parabolic_noises)
        figs = []
        for slope in multipart_parabolic_slopes:
            for noise, c in zip(multipart_parabolic_noises, colors):
                fig, ax = plt.subplots()
                figs.append(fig)
                x, y = f(slope, noise, unsort=False, na_data=False)
                x0, y0 = f(slope, 0, False, False)
                ax.plot(x0, y0, c='k', label='no noise', ls=':', alpha=0.5)
                ax.scatter(x, y, label=f'noise:{noise}', c=c)
                ax.legend()
                ax.set_title(f'slope:{slope}, f:{f.__name__}')
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_plot_seasonal_multipart_para(self, show=False):
        figs = []
        f = make_seasonal_multipart_parabolic
        for slope in multipart_parabolic_slopes:
            for noise in multipart_parabolic_noises:
                fig, ax = plt.subplots()
                figs.append(fig)
                data = f(slope, noise, unsort=False, na_data=False)
                ax.scatter(data.index, data.y, label=f'noise:{noise}', c=data.seasons)
                ax.legend()
                ax.set_title(f'slope:{slope}, f:{f.__name__}')
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_plot_seasonal_multipart_sharp(self, show=False):
        figs = []
        f = make_seasonal_multipart_sharp_change
        for slope in multipart_sharp_slopes:
            for noise in multipart_sharp_noises:
                fig, ax = plt.subplots()
                figs.append(fig)
                data = f(slope, noise, unsort=False, na_data=False)
                ax.scatter(data.index, data.y, label=f'noise:{noise}', c=data.seasons)
                ax.legend()
                ax.set_title(f'slope:{slope}, f:{f.__name__}')
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_generate_startpoints(self):
        save_path = Path(__file__).parent.joinpath('test_data', 'test_generate_startpoints.npz')
        write_test_data = False

        x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                unsort=False, na_data=False)
        part4 = _generate_startpoints(n=len(x), min_size=10, nparts=4, test=True)
        part3 = _generate_startpoints(n=len(x), min_size=10, nparts=3, test=True)
        part2 = _generate_startpoints(n=len(x), min_size=10, nparts=2, test=True)

        # test check_step
        part4_check_step = _generate_startpoints(n=len(x), min_size=10, nparts=4, test=True, check_step=2)
        part3_check_step = _generate_startpoints(n=len(x), min_size=10, nparts=3, test=True, check_step=2)
        part2_check_step = _generate_startpoints(n=len(x), min_size=10, nparts=2, test=True, check_step=2)

        #  test check_window
        part4_check_window = _generate_startpoints(n=len(x), min_size=10, nparts=4, test=True,
                                                   check_window=[
                                                       (10, 20),
                                                       (40, 50),
                                                       (60, 70),
                                                   ])
        part3_check_window = _generate_startpoints(n=len(x), min_size=10, nparts=3, test=True,
                                                   check_window=[
                                                       (20, 50),
                                                       (50, 70),

                                                   ])
        part2_check_window = _generate_startpoints(n=len(x), min_size=10, nparts=2, test=True, check_window=(20, 50))

        # check both check_step and check_window
        part4_check_window_step = _generate_startpoints(n=len(x), min_size=10, nparts=4, test=True,
                                                        check_window=[
                                                            (10, 20),
                                                            (40, 50),
                                                            (60, 70),
                                                        ], check_step=3)
        part3_check_window_step = _generate_startpoints(n=len(x), min_size=10, nparts=3, test=True,
                                                        check_window=[
                                                            (20, 50),
                                                            (50, 70),

                                                        ], check_step=3)
        part2_check_window_step = _generate_startpoints(n=len(x), min_size=10, nparts=2, test=True,
                                                        check_window=(20, 50),
                                                        check_step=3)

        if write_test_data:
            np.savez_compressed(save_path, part4=part4, part3=part3, part2=part2,
                                part4_check_step=part4_check_step,
                                part3_check_step=part3_check_step,
                                part2_check_step=part2_check_step,
                                part4_check_window=part4_check_window,
                                part3_check_window=part3_check_window,
                                part2_check_window=part2_check_window,
                                part4_check_window_step=part4_check_window_step,
                                part3_check_window_step=part3_check_window_step,
                                part2_check_window_step=part2_check_window_step,

                                )
        expect = np.load(save_path)
        check_keys = [
            'part4', 'part3', 'part2',
            'part4_check_step', 'part3_check_step', 'part2_check_step',
            'part4_check_window', 'part3_check_window', 'part2_check_window',
            'part4_check_window_step', 'part3_check_window_step', 'part2_check_window_step',
        ]
        for k in check_keys:
            self.assertTrue(np.allclose(eval(k), expect[k]), f'{k} failed')

    def test_multipart_plotting(self, show=False):
        # 2 part
        figs = []
        x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                unsort=False, na_data=False)
        data = pd.Series(y, index=x)
        mk = MultiPartKendall(data=data, data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                              nparts=2, expect_part=(1, -1), min_size=10,
                              serialise_path=None, recalc=False, initalize=True)

        fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
        figs.append(fig)
        fig, ax = mk.plot_data_from_breakpoints(50, ax=axs[0])
        fig, ax = mk.plot_data_from_breakpoints(25, ax=axs[1], txt_vloc=0.01)
        fig.tight_layout()

        x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                unsort=False, na_data=False)
        data = pd.Series(y, index=datetime.datetime.today() + pd.to_timedelta(x * 7, unit='D'))

        mk = MultiPartKendall(data=data, data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                              nparts=2, expect_part=(1, -1), min_size=10,
                              serialise_path=None, recalc=False, initalize=True)

        fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
        figs.append(fig)
        fig, ax = mk.plot_data_from_breakpoints(50, ax=axs[0])
        mk.plot_data_from_breakpoints(25, ax=axs[1], txt_vloc=0.01)
        fig.tight_layout()

        # 3 part
        x, y = make_multipart_parabolic_data(slope=multipart_parabolic_slopes[0], noise=multipart_parabolic_noises[1],
                                             unsort=False, na_data=False)
        data = pd.Series(y, index=x)
        mk_para = MultiPartKendall(data=data, data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                   nparts=3, expect_part=(1, 0, -1), min_size=10,
                                   serialise_path=None, recalc=False, initalize=True)
        # plot
        fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
        figs.append(fig)
        fig, ax = mk_para.plot_data_from_breakpoints([40, 60], ax=axs[0])
        mk_para.plot_data_from_breakpoints([50, 60], ax=axs[1], txt_vloc=+0.1)
        fig.tight_layout()

        # plot acceptable matches
        for k in ['p', 'z', 's', 'var_s']:
            print(k)
            fig, ax = mk_para.plot_acceptable_matches(key=k)
            figs.append(fig)
            ax.set_title('para')
            fig, ax = mk.plot_acceptable_matches(key=k)
            figs.append(fig)
            ax.set_title('sharp')
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_multipart_serialisation(self):
        with tempfile.TemporaryDirectory() as tdir:
            # 2 part
            tdir = Path(tdir)
            x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                    unsort=False, na_data=False)
            data = pd.Series(y, index=x)
            mk = MultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10,
                                  data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                  serialise_path=tdir.joinpath('test2.hdf'), recalc=False, initalize=True)

            mk1 = MultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10,
                                   data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                   serialise_path=tdir.joinpath('test2.hdf'), recalc=False, initalize=True)

            mk2 = MultiPartKendall.from_file(tdir.joinpath('test2.hdf'))

            self.assertEqual(mk, mk1)
            self.assertEqual(mk, mk2)

            # 3part
            x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                    unsort=False, na_data=False)
            data = pd.Series(y, index=x)
            mk = MultiPartKendall(data=data, nparts=3, expect_part=(1, 0, -1), min_size=10,
                                  data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                  serialise_path=tdir.joinpath('test3.hdf'), recalc=False, initalize=True)

            mk1 = MultiPartKendall(data=data, nparts=3, expect_part=(1, 0, -1), min_size=10,
                                   data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                   serialise_path=tdir.joinpath('test3.hdf'), recalc=False, initalize=True)

            mk2 = MultiPartKendall.from_file(tdir.joinpath('test3.hdf'))

            self.assertEqual(mk, mk1)
            self.assertEqual(mk, mk2)

    def test_seasonal_multipart_serialisation(self):
        with tempfile.TemporaryDirectory() as tdir:
            # 2 part
            tdir = Path(tdir)
            data = make_seasonal_multipart_sharp_change(slope=multipart_sharp_slopes[0],
                                                        noise=multipart_sharp_noises[1],
                                                        unsort=False, na_data=False)
            mk = SeasonalMultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10,
                                          data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                          no_trend_alpha=0.5,
                                          serialise_path=tdir.joinpath('test2.hdf'), recalc=False, initalize=True)

            mk1 = SeasonalMultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10,
                                           data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                           no_trend_alpha=0.5,
                                           serialise_path=tdir.joinpath('test2.hdf'), recalc=False, initalize=True)

            mk2 = SeasonalMultiPartKendall.from_file(tdir.joinpath('test2.hdf'))

            self.assertEqual(mk, mk1)
            self.assertEqual(mk, mk2)

            # 3part
            data = make_seasonal_multipart_sharp_change(slope=multipart_sharp_slopes[0],
                                                        noise=multipart_sharp_noises[1],
                                                        unsort=False, na_data=False)
            mk = SeasonalMultiPartKendall(data=data, nparts=3, expect_part=(1, 0, -1), min_size=10,
                                          data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                          no_trend_alpha=0.5,
                                          serialise_path=tdir.joinpath('test3.hdf'), recalc=False, initalize=True)

            mk1 = SeasonalMultiPartKendall(data=data, nparts=3, expect_part=(1, 0, -1), min_size=10,
                                           data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                           no_trend_alpha=0.5,
                                           serialise_path=tdir.joinpath('test3.hdf'), recalc=False, initalize=True)

            mk2 = SeasonalMultiPartKendall.from_file(tdir.joinpath('test3.hdf'))

            self.assertEqual(mk, mk1)
            self.assertEqual(mk, mk2)

    def test_seasonal_multipart_plotting(self, show=False):
        figs = []
        data = make_seasonal_multipart_sharp_change(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                    unsort=False, na_data=False)
        mk = SeasonalMultiPartKendall(data=data, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                      no_trend_alpha=0.5,
                                      nparts=2, expect_part=(1, -1), min_size=10,
                                      serialise_path=None, recalc=False, initalize=True)

        fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
        figs.append(fig)
        fig, ax = mk.plot_data_from_breakpoints(50, ax=axs[0])
        mk.plot_data_from_breakpoints(25, ax=axs[1], txt_vloc=0.01)
        fig.tight_layout()

        # 3 part
        data = make_seasonal_multipart_parabolic(slope=multipart_parabolic_slopes[0],
                                                 noise=multipart_parabolic_noises[1],
                                                 unsort=False, na_data=False)
        mk_para = SeasonalMultiPartKendall(data=data, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                           no_trend_alpha=0.5,
                                           nparts=3, expect_part=(1, 0, -1), min_size=10,
                                           serialise_path=None, recalc=False, initalize=True)
        # plot
        fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
        figs.append(fig)
        fig, ax = mk_para.plot_data_from_breakpoints([40, 60], ax=axs[0])
        mk_para.plot_data_from_breakpoints([50, 60], ax=axs[1], txt_vloc=+0.1)
        fig.tight_layout()

        # plot acceptable matches
        for k in ['p', 'z', 's', 'var_s']:
            print(k)
            fig, ax = mk_para.plot_acceptable_matches(key=k)
            figs.append(fig)
            ax.set_title('para')
            fig, ax = mk.plot_acceptable_matches(key=k)
            figs.append(fig)
            ax.set_title('sharp')
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_multipart_kendall(self, show=False, print_total=False):
        """

        :param show: flag, show the plots of the data
        :param print_total: flag, just count number of iterations
        :return:
        """
        figs = []
        write_test_data = False
        make_functions = [
            make_multipart_parabolic_data,
            make_multipart_sharp_change_data,
        ]
        make_names = [
            'para',
            'sharp',
        ]
        iter_datas = [
            [
                multipart_parabolic_slopes,
                [multipart_parabolic_noises[1], multipart_parabolic_noises[-1]],
                [False],
                [False],
            ],
            [
                multipart_sharp_slopes,
                [multipart_sharp_noises[1], multipart_sharp_noises[-1]],
                [False, True],
                [False, True],
            ],
        ]

        npart_data = [
            (3, (1, 0, -1), [40, 60]),
            (2, (1, -1), [50]),
        ]
        total = 0
        for mfunc, mname, iterdata, (npart, epart, bpoints) in zip(make_functions, make_names, iter_datas, npart_data):
            for slope, noise, unsort, na_data in itertools.product(*iterdata):
                total += 1
                if print_total:
                    continue
                title = f'mk_{mname}, {npart=}\n'
                title += f'{slope=}, {noise=}, {unsort=}, {na_data=}\n'
                print(title)
                x, y = mfunc(slope=slope, noise=noise, unsort=unsort, na_data=na_data)
                data = pd.Series(y, index=x)
                if slope != 0:
                    use_epart = [e * np.sign(slope) for e in epart]
                else:
                    use_epart = epart
                mk = MultiPartKendall(data, nparts=npart,
                                      expect_part=use_epart,
                                      min_size=10,
                                      alpha=0.05, no_trend_alpha=0.5,
                                      data_col=None, rm_na=True,
                                      serialise_path=None)
                title += f'breakpoint: {mk.idx_values[bpoints]}'
                fname = f'mk_{mname}_' + '_'.join([str(i) for i in [
                    slope, noise, unsort, na_data]]).replace('.', '_').replace('-', '_')
                fname += f'_npart{npart}.hdf'
                accept = mk.get_acceptable_matches().astype(float)
                bpoint_data, kstats = mk.get_data_from_breakpoints(bpoints)
                hdf_path = Path(__file__).parent.joinpath('test_data', fname)
                if write_test_data:
                    mk.to_file(hdf_path)
                    accept.to_hdf(hdf_path, 'accept_data', complevel=9, complib='blosc:lz4')
                    for i, df in enumerate(bpoint_data):
                        df.to_hdf(hdf_path, f'bp_data_{i}', complevel=9, complib='blosc:lz4')
                    fig, ax = mk.plot_data_from_breakpoints(breakpoints=bpoints)
                    figs.append(fig)
                    ax.set_title(title)
                    if show:
                        plt.show()
                    plt.close('all')
                else:
                    mk1 = MultiPartKendall.from_file(hdf_path)
                    accept1 = pd.read_hdf(hdf_path, 'accept_data')
                    bpoint_data1 = []
                    for i in range(npart):
                        bpoint_data1.append(pd.read_hdf(hdf_path, f'bp_data_{i}'))
                    self.assertIsInstance(accept1, pd.DataFrame)
                    self.assertEqual(mk, mk1)
                    self.assertNotEqual(id(mk), id(mk1))
                    for i in range(npart):
                        pd.testing.assert_frame_equal(pd.DataFrame(bpoint_data[i]), pd.DataFrame(bpoint_data1[i]))

        print(f'total tests: {total}')

        # test sorting vs unsorting doesnt change anything
        for slope, noise, unsort, na_data in itertools.product(*iter_datas[1]):
            if not unsort:
                continue
            fname_unsort = f'mk_sharp_' + '_'.join([str(i) for i in [
                slope, noise, unsort, na_data]]).replace('.', '_').replace('-', '_')
            fname_unsort += f'_npart2.hdf'

            fname_sort = f'mk_sharp_' + '_'.join([str(i) for i in [
                slope, noise, False, na_data]]).replace('.', '_').replace('-', '_')
            fname_sort += f'_npart2.hdf'

            mk_sort = MultiPartKendall.from_file(Path(__file__).parent.joinpath('test_data', fname_sort))
            mk_unsort = MultiPartKendall.from_file(Path(__file__).parent.joinpath('test_data', fname_unsort))
            self.assertEqual(mk_sort, mk_unsort)
        for fig in figs:
            plt.close(fig)

    def test_seasonal_multipart_kendall(self, show=False, print_total=False):
        figs = []
        write_test_data = False
        make_functions = [
            make_seasonal_multipart_sharp_change,
            make_seasonal_multipart_parabolic,
        ]
        make_names = [
            'sharp',
            'para',
        ]
        iter_datas = [
            [
                multipart_sharp_slopes,
                [multipart_sharp_noises[1], multipart_sharp_noises[-1]],
                [True],
                [True],
            ],
            [
                multipart_parabolic_slopes,
                [multipart_parabolic_noises[1], multipart_parabolic_noises[-1]],
                [False],
                [False],
            ],
        ]

        npart_data = [
            (2, (1, -1), [50]),
            (3, (1, 0, -1), [40, 60]),
        ]
        total = 0
        for mfunc, mname, iterdata, (npart, epart, bpoints) in zip(make_functions, make_names, iter_datas, npart_data):
            for slope, noise, unsort, na_data in itertools.product(*iterdata):
                total += 1
                if print_total:
                    continue
                title = f'mk_{mname}, {npart=}\n'
                title += f'{slope=}, {noise=}, {unsort=}, {na_data=}\n'
                print(title)
                data = mfunc(slope=slope, noise=noise, unsort=unsort, na_data=na_data)
                if slope != 0:
                    use_epart = [e * np.sign(slope) for e in epart]
                else:
                    use_epart = epart
                mk = SeasonalMultiPartKendall(data, nparts=npart,
                                              expect_part=use_epart,
                                              min_size=10,
                                              alpha=0.05, no_trend_alpha=0.5,
                                              data_col='y', season_col='seasons', rm_na=True,
                                              serialise_path=None)
                title += f'breakpoint: {mk.idx_values[bpoints]}'
                fname = f'smk_{mname}_' + '_'.join([str(i) for i in [
                    slope, noise, unsort, na_data]]).replace('.', '_').replace('-', '_')
                fname += f'_npart{npart}.hdf'
                accept = mk.get_acceptable_matches().astype(float)
                bpoint_data, kstats = mk.get_data_from_breakpoints(bpoints)
                hdf_path = Path(__file__).parent.joinpath('test_data', fname)
                if write_test_data:
                    mk.to_file(hdf_path)
                    accept.to_hdf(hdf_path, 'accept_data', complevel=9, complib='blosc:lz4')
                    for i, df in enumerate(bpoint_data):
                        df.to_hdf(hdf_path, f'bp_data_{i}', complevel=9, complib='blosc:lz4')
                    fig, ax = mk.plot_data_from_breakpoints(breakpoints=bpoints)
                    figs.append(fig)
                    ax.set_title(title)
                    if show:
                        plt.show()
                    plt.close('all')
                else:
                    mk1 = SeasonalMultiPartKendall.from_file(hdf_path)
                    accept1 = pd.read_hdf(hdf_path, 'accept_data')
                    bpoint_data1 = []
                    for i in range(npart):
                        bpoint_data1.append(pd.read_hdf(hdf_path, f'bp_data_{i}'))
                    self.assertIsInstance(accept1, pd.DataFrame)
                    self.assertEqual(mk, mk1)
                    for i in range(npart):
                        pd.testing.assert_frame_equal(pd.DataFrame(bpoint_data[i]), pd.DataFrame(bpoint_data1[i]))

        print(f'total tests: {total}')
        for fig in figs:
            plt.close(fig)

    def test_get_best_data(self, show=False):
        figs = []
        x_para, y_para = make_multipart_parabolic_data(slope=multipart_parabolic_slopes[0],
                                                       noise=0,
                                                       unsort=False,
                                                       na_data=False)
        data = pd.Series(index=x_para, data=y_para)
        mk = MultiPartKendall(
            data=data,  # data can be passed as a np.array, pd.Series, or pd.DataFrame
            nparts=3,  # number of parts to split data into
            expect_part=(1, 0, -1),  # the expected slope of each part (1, increasing, 0, no change, -1, decreasing)
            min_size=10,
            data_col=None,
            alpha=0.05,  # significance level for trends (p<alpha)
            no_trend_alpha=0.5,  # significance level for no trend (p>no_trend_alpha)
            rm_na=True,
            serialise_path=None,  # None or path to serialise results to
            recalc=False)
        fig, ax = mk.plot_acceptable_matches(key='znorm_joint')
        figs.append(fig)
        best = mk.get_maxz_breakpoints()
        self.assertTrue(all([best[i] == [(44, 55)][i] for i in range(len(best))]))
        ax.set_title(f'para: {best=}')

        x_sharp, y_sharp = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0],
                                                            noise=0,
                                                            unsort=False,
                                                            na_data=False)
        data = pd.Series(index=x_sharp, data=y_sharp)
        mk = MultiPartKendall(
            data=data,  # data can be passed as a np.array, pd.Series, or pd.DataFrame
            nparts=2,  # number of parts to split data into
            expect_part=(1, -1),  # the expected slope of each part (1, increasing, 0, no change, -1, decreasing)
            min_size=10,
            data_col=None,
            alpha=0.05,  # significance level for trends (p<alpha)
            no_trend_alpha=0.5,  # significance level for no trend (p>no_trend_alpha)
            rm_na=True,
            serialise_path=None,  # None or path to serialise results to
            recalc=False)
        fig, ax = mk.plot_acceptable_matches(key='znorm_joint')
        figs.append(fig)
        best = mk.get_maxz_breakpoints()
        self.assertEqual(best, 50)
        ax.set_title(f'sharp: {best=}')
        if show:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_check_step_window_mpmk(self):
        save_paths = {}
        names = ['mk_check_step', 'mk_check_window', 'mk_check_window_step']
        for name in names:
            save_paths[name] = Path(__file__).parent.joinpath('test_data', f'{name}.hdf')
        write_test_data = False
        x, data = make_multipart_parabolic_data(slope=multipart_parabolic_slopes[0],
                                                noise=multipart_parabolic_noises[1],
                                                unsort=False, na_data=False)

        org_mk = MultiPartKendall(data, nparts=3,
                                  expect_part=(1, 0, -1),
                                  min_size=10,
                                  alpha=0.05, no_trend_alpha=0.5,
                                  data_col=None, rm_na=True,
                                  serialise_path=None,
                                  check_step=1,
                                  check_window=None,
                                  )

        mk_check_step = MultiPartKendall(data, nparts=3,
                                         expect_part=(1, 0, -1),
                                         min_size=10,
                                         alpha=0.05, no_trend_alpha=0.5,
                                         data_col=None, rm_na=True,
                                         serialise_path=None,
                                         check_step=3,
                                         check_window=None,
                                         )

        mk_check_window = MultiPartKendall(data, nparts=3,
                                           expect_part=(1, 0, -1),
                                           min_size=10,
                                           alpha=0.05, no_trend_alpha=0.5,
                                           data_col=None, rm_na=True,
                                           serialise_path=None,
                                           check_step=1,
                                           check_window=[(15, 30), (60, 80)],
                                           )

        mk_check_window_step = MultiPartKendall(data, nparts=3,
                                                expect_part=(1, 0, -1),
                                                min_size=10,
                                                alpha=0.05, no_trend_alpha=0.5,
                                                data_col=None, rm_na=True,
                                                serialise_path=None,
                                                check_step=3,
                                                check_window=[(15, 30), (60, 80)],
                                                )

        # check serialisation/ against saved data
        if write_test_data:
            for k, op in save_paths.items():
                t = eval(k)
                self.assertIsInstance(t, MultiPartKendall)
                t.to_file(op)

        for k, op in save_paths.items():
            t = eval(k)
            self.assertIsInstance(t, MultiPartKendall)
            t1 = MultiPartKendall.from_file(op)
            self.assertEqual(t, t1)

        # check results compared to orignial
        org_accept = org_mk.get_acceptable_matches()
        mk_check_step_accept = mk_check_step.get_acceptable_matches()
        mk_check_window_accept = mk_check_window.get_acceptable_matches()
        mk_check_window_step_accept = mk_check_window_step.get_acceptable_matches()

        for df in [mk_check_step_accept, mk_check_window_accept, mk_check_window_step_accept]:
            self.assertIsInstance(df, pd.DataFrame)
            temp = org_accept.loc[df.index]
            check_cols = [c for c in df.columns if 'znorm' not in c]
            self.assertEqual(len(check_cols), 6 * 3)
            pd.testing.assert_frame_equal(temp[check_cols], df[check_cols])

    def test_check_step_window_smpmk(self):
        save_paths = {}
        names = ['smk_check_step', 'smk_check_window', 'smk_check_window_step']
        for name in names:
            save_paths[name] = Path(__file__).parent.joinpath('test_data', f'{name}.hdf')
        write_test_data = False
        data = make_seasonal_multipart_parabolic(slope=multipart_parabolic_slopes[0],
                                                 noise=multipart_parabolic_noises[1],
                                                 unsort=False, na_data=False)

        org_smk = SeasonalMultiPartKendall(data, nparts=3,
                                           expect_part=(1, 0, -1),
                                           min_size=10,
                                           alpha=0.05, no_trend_alpha=0.5,
                                           data_col='y', season_col='seasons', rm_na=True,
                                           serialise_path=None,
                                           check_step=1,
                                           check_window=None,
                                           )

        smk_check_step = SeasonalMultiPartKendall(data, nparts=3,
                                                  expect_part=(1, 0, -1),
                                                  min_size=10,
                                                  alpha=0.05, no_trend_alpha=0.5,
                                                  data_col='y', season_col='seasons', rm_na=True,
                                                  serialise_path=None,
                                                  check_step=3,
                                                  check_window=None,
                                                  )

        smk_check_window = SeasonalMultiPartKendall(data, nparts=3,
                                                    expect_part=(1, 0, -1),
                                                    min_size=10,
                                                    alpha=0.05, no_trend_alpha=0.5,
                                                    data_col='y', season_col='seasons', rm_na=True,
                                                    serialise_path=None,
                                                    check_step=1,
                                                    check_window=[(15, 30), (60, 80)],
                                                    )

        smk_check_window_step = SeasonalMultiPartKendall(data, nparts=3,
                                                         expect_part=(1, 0, -1),
                                                         min_size=10,
                                                         alpha=0.05, no_trend_alpha=0.5,
                                                         data_col='y', season_col='seasons', rm_na=True,
                                                         serialise_path=None,
                                                         check_step=3,
                                                         check_window=[(15, 30), (60, 80)],
                                                         )

        # check serialisation/ against saved data
        if write_test_data:
            for k, op in save_paths.items():
                t = eval(k)
                self.assertIsInstance(t, SeasonalMultiPartKendall)
                t.to_file(op)

        for k, op in save_paths.items():
            t = eval(k)
            self.assertIsInstance(t, SeasonalMultiPartKendall)
            t1 = SeasonalMultiPartKendall.from_file(op)
            self.assertIsInstance(t1, SeasonalMultiPartKendall)

            self.assertTrue(t, t1)

        # check results compared to orignial
        org_accept = org_smk.get_acceptable_matches()
        smk_check_step_accept = smk_check_step.get_acceptable_matches()
        smk_check_window_accept = smk_check_window.get_acceptable_matches()
        smk_check_window_step_accept = smk_check_window_step.get_acceptable_matches()

        for df in [smk_check_step_accept, smk_check_window_accept, smk_check_window_step_accept]:
            self.assertIsInstance(df, pd.DataFrame)
            temp = org_accept.loc[df.index]
            check_cols = [c for c in df.columns if 'znorm' not in c]
            self.assertEqual(len(check_cols), 6 * 3)
            pd.testing.assert_frame_equal(temp[check_cols], df[check_cols])


if __name__ == '__main__':
    unittest.main()
