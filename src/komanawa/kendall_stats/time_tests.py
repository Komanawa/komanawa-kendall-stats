"""
usage python time_tests.py [outdir]
:param outdir: path to save the results to, if not provided then the results are saved to the same directory as the script

created matt_dumont 
on: 29/09/23
"""
import pandas as pd

from komanawa.kendall_stats import MultiPartKendall, SeasonalMultiPartKendall, SeasonalKendall, MannKendall
from komanawa.kendall_stats import example_data

from pathlib import Path
import timeit
import sys
import os


def timeit_test(function_names, npoints, n=10):
    """
    run an automated timeit test, must be outside of the function definition, prints results in scientific notation
    units are seconds

    :param py_file_path: path to the python file that holds the functions, if the functions are in the same script as call then  __file__ is sufficient. in this case the function call should be protected by: if __name__ == '__main__':
    :param function_names: the names of the functions to test (iterable), functions must not have arguments
    :param n: number of times to test
    :return: dictionary of function names and their times
    """
    py_file_path = __file__
    print(py_file_path)
    d = os.path.dirname(py_file_path)
    fname = os.path.basename(py_file_path).replace('.py', '')
    sys.path.append(d)

    out = {}
    for fn in function_names:
        print(f'testing: {fn}({npoints})')
        t = timeit.timeit(f'{fn}({npoints})',
                          setup='from {} import {}'.format(fname, fn),
                          number=n) / n
        out[fn] = t
        print('{0:e} seconds'.format(t))
    return out


def MannKendall_time_test(npoints):
    npoints = int(npoints)
    x, y = make_example_data.make_increasing_decreasing_data(slope=0.1, noise=5, step=100 / npoints)
    MannKendall(y)


def SeasonalKendall_time_test(npoints):
    npoints = int(npoints)
    data = make_example_data.make_seasonal_data(slope=0.1, noise=5, unsort=False, na_data=False, step=100 / npoints)
    SeasonalKendall(df=data, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                    freq_limit=1)


def MultiPartKendall_2part_time_test(npoints):
    npoints = int(npoints)
    x, y = make_example_data.make_multipart_sharp_change_data(slope=make_example_data.multipart_sharp_slopes[0],
                                                              noise=make_example_data.multipart_sharp_noises[0],
                                                              unsort=False, na_data=False, step=100 / npoints)
    t = MultiPartKendall(data=y,
                         nparts=2, expect_part=(1, -1),
                         min_size=10,
                         alpha=0.05, no_trend_alpha=0.5,
                         data_col=None, rm_na=True,
                         serialise_path=None, recalc=False, )
    t.get_maxz_breakpoints()


def SeasonalMultiPartKendall_2part_time_test(npoints):
    npoints = int(npoints)
    data = make_example_data.make_seasonal_multipart_sharp_change(slope=make_example_data.multipart_sharp_slopes[0],
                                                                  noise=make_example_data.multipart_sharp_noises[0],
                                                                  unsort=False, na_data=False, step=100 / npoints)
    t = SeasonalMultiPartKendall(data, data_col='y', season_col='seasons',
                                 nparts=2, expect_part=(1, -1), min_size=10,
                                 alpha=0.05, no_trend_alpha=0.5,
                                 rm_na=True,
                                 serialise_path=None, freq_limit=1, recalc=False, initalize=True)
    t.get_maxz_breakpoints()


def MultiPartKendall_3part_time_test(npoints):
    npoints = int(npoints)
    x, y = make_example_data.make_multipart_parabolic_data(slope=make_example_data.multipart_parabolic_slopes[0],
                                                           noise=make_example_data.multipart_parabolic_noises[0],
                                                           unsort=False, na_data=False, step=100 / npoints)
    t = MultiPartKendall(y, data_col=None,
                         nparts=3, expect_part=(1, 0, -1), min_size=10,
                         alpha=0.05, no_trend_alpha=0.5,
                         rm_na=True,
                         serialise_path=None, recalc=False)
    t.get_maxz_breakpoints()


def SeasonalMultiPartKendall_3part_time_test(npoints):
    npoints = int(npoints)
    data = make_example_data.make_seasonal_multipart_parabolic(slope=make_example_data.multipart_parabolic_slopes[0],
                                                               noise=make_example_data.multipart_parabolic_noises[0],
                                                               unsort=False, na_data=False, step=100 / npoints)
    t = SeasonalMultiPartKendall(data, data_col='y', season_col='seasons',
                                 nparts=3, expect_part=(1, 0, -1), min_size=10,
                                 alpha=0.05, no_trend_alpha=0.5,
                                 rm_na=True,
                                 serialise_path=None, freq_limit=1, recalc=False, initalize=True)
    t.get_maxz_breakpoints()


def test_all_functions():
    use_npoints = '50'
    MannKendall_time_test(use_npoints)
    SeasonalKendall_time_test(use_npoints)
    MultiPartKendall_2part_time_test(use_npoints)
    SeasonalMultiPartKendall_2part_time_test(use_npoints)
    MultiPartKendall_3part_time_test(use_npoints)
    SeasonalMultiPartKendall_3part_time_test(use_npoints)


def run_time_test(outdir=None, all_npoints=['50', '100', '500', '1000'],
                  function_names=['MannKendall_time_test', 'SeasonalKendall_time_test',
                                  'MultiPartKendall_2part_time_test', 'SeasonalMultiPartKendall_2part_time_test',
                                  'MultiPartKendall_3part_time_test', 'SeasonalMultiPartKendall_3part_time_test']
                  ):
    """
    run the time test for all functions and save the results to a csv file

    :param outdir: place to save the output
    :param all_npoints: the dataset sizes to test
    :param function_names: the names of the functions to test, default is all
    :return:
    """
    assert set(function_names).issubset(['MannKendall_time_test',
                                         'SeasonalKendall_time_test',
                                         'MultiPartKendall_2part_time_test',
                                         'SeasonalMultiPartKendall_2part_time_test',
                                         'MultiPartKendall_3part_time_test',
                                         'SeasonalMultiPartKendall_3part_time_test'])
    if outdir is None:
        outdir = Path(__file__).parent
    else:
        outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    outdata = pd.DataFrame(index=all_npoints, columns=function_names)
    outdata.index.name = 'npoints'
    for npoints in all_npoints:
        if int(npoints) > 500:
            use_n = 1

        else:
            use_n = 2
        print(f'testing {npoints}')
        temp = timeit_test(function_names, npoints, n=use_n)
        outdata.loc[npoints] = pd.Series(temp)
    print(f'saving results to {outdir.joinpath("time_test_results.txt")}')
    outdata.to_csv(outdir.joinpath('time_test_results.txt'))


if __name__ == '__main__':
    args = sys.argv
    outdir = None
    if len(args) > 1:
        outdir = args[1]
    run_time_test(outdir)
