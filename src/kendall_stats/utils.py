"""
created matt_dumont 
on: 29/09/23
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from pathlib import Path


def estimate_runtime(npoints, func, plot=False):
    """
    assumes log-linear relationship between runtime and number of points
    :param npoints:
    :param func:
    :param plot: if True then plot the data and the regression line
    :return:
    """
    assert func in ['MannKendall', 'SeasonalKendall', 'MultiPartKendall_2part', 'SeasonalMultiPartKendall_2part',
                        'MultiPartKendall_3part', 'SeasonalMultiPartKendall_3part']

    data = pd.read_csv(Path(__file__).parent.joinpath('time_test_results.txt'), index_col=0)
    data.columns = [e.replace('_time_test','') for e in data.columns]
    use_data = data[func]
    lr = linregress(use_data.index, np.log10(use_data))
    out = 10 ** (lr.intercept + lr.slope * npoints)
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(use_data.index, use_data, c='b', label='data')
        x = np.arange(10, np.max([use_data.index, npoints]))
        ax.plot(x, 10 ** (lr.intercept + lr.slope * x))

        ax.set_yscale('log')
        ax.set_title(f'{func} runtime estimate in seconds')
        plt.show()
    return out

if __name__ == '__main__':
    for f in ['MannKendall', 'SeasonalKendall', 'MultiPartKendall_2part', 'SeasonalMultiPartKendall_2part',
                        'MultiPartKendall_3part', 'SeasonalMultiPartKendall_3part']:
        print(f, estimate_runtime(np.array([500, 1000,5000,10000]), f, plot=False))
