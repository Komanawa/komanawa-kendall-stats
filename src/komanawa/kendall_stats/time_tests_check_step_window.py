"""
usage python time_tests.py [outdir]
:param outdir: path to save the results to, if not provided then the results are saved
               to the same directory as the script

created matt_dumont 
on: 29/09/23
"""
import itertools

import pandas as pd

from komanawa.kendall_stats import MultiPartKendall, SeasonalMultiPartKendall
from komanawa.kendall_stats import make_example_data
from pathlib import Path
import timeit
import sys
import os


def timeit_test(function_names, npoints, check_step, check_window, n=10, ):
    """
    run an automated timeit test, must be outside of the function definition, prints results in scientific notation
    units are seconds

    :param py_file_path: path to the python file that holds the functions,
                        if the functions are in the same script as call then  __file__ is sufficient.
                        in this case the function call should be protected by: if __name__ == '__main__':
    :param function_names: the names of the functions to test (iterable), functions must not have arguments
    :param n: number of times to test
    :return:
    """
    py_file_path = __file__
    print(py_file_path)
    d = os.path.dirname(py_file_path)
    fname = os.path.basename(py_file_path).replace('.py', '')
    sys.path.append(d)

    out = {}
    for fn in function_names:
        print(f'testing: {fn}({npoints})')
        t = timeit.timeit(f'{fn}({npoints},{check_step}, {check_window})',
                          setup='from {} import {}'.format(fname, fn),
                          number=n) / n
        out[fn] = t
        print('{0:e} seconds'.format(t))
    return out


def MultiPartKendall_2part_time_test(npoints, check_step, check_window):
    check_step = int(check_step)
    npoints = int(npoints)
    x, y = make_example_data.make_multipart_sharp_change_data(slope=make_example_data.multipart_sharp_slopes[0],
                                                              noise=make_example_data.multipart_sharp_noises[0],
                                                              unsort=False, na_data=False, step=100 / npoints)
    t = MultiPartKendall(data=y,
                         nparts=2, expect_part=(1, -1),
                         min_size=10,
                         alpha=0.05, no_trend_alpha=0.5,
                         data_col=None, rm_na=True,
                         check_step=check_step, check_window=check_window,
                         serialise_path=None, recalc=False, )
    t.get_maxz_breakpoints()


def run_time_test(outdir=None):
    if outdir is None:
        outdir = Path(__file__).parent
    else:
        outdir = Path(outdir)
    check_steps = ['1', '2', '5', '10']
    check_windows = ['None', '(300, 700)', '(400, 600)', '(450,550)', '(475,525)']
    outdir.mkdir(exist_ok=True, parents=True)
    function_names = ['MultiPartKendall_2part_time_test']
    outdata = pd.DataFrame()
    for cs, cw in itertools.product(check_steps, check_windows):
        print(f'testing {cs}, {cw}')
        temp = timeit_test(function_names, npoints=1000, check_step=cs, check_window=cw, n=1)
        outdata.loc[f'step: {cs}', f'window: {str(cw)}'] = f"{temp['MultiPartKendall_2part_time_test']:.4e}"
    print(f'saving results to {outdir.joinpath("time_test_check_step_window_results.txt")}')
    outdata.to_csv(outdir.joinpath('time_test_check_step_window_results.txt'))


if __name__ == '__main__':
    args = sys.argv
    outdir = None
    if len(args) > 1:
        outdir = args[1]
    run_time_test(outdir)
