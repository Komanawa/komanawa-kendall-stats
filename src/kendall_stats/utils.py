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
    assumes linear log-log relationship between runtime and number of points
    :param npoints:
    :param func:
    :param plot: if True then plot the data and the regression line
    :return:
    """
    assert func in ['MannKendall', 'SeasonalKendall', 'MultiPartKendall_2part', 'SeasonalMultiPartKendall_2part',
                        'MultiPartKendall_3part', 'SeasonalMultiPartKendall_3part']

    data = pd.read_csv(Path(__file__).parent.joinpath('time_test_results.txt'), index_col=0)
    data.columns = [e.replace('_time_test','') for e in data.columns]
    use_data = data[func].dropna()
    lr = linregress(np.log10(use_data.index), np.log10(use_data))
    out = 10 ** (lr.intercept + lr.slope * np.log10(npoints))
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(use_data.index, use_data, c='b', label='data')
        x = np.arange(10, np.max(np.concatenate([use_data.index, npoints])))
        ax.plot(x, 10 ** (lr.intercept + lr.slope * np.log10(x)), c='k', label='regression', ls='--')
        use_y = 10 ** (lr.intercept + lr.slope * np.log10(npoints))
        ax.scatter(npoints, use_y, c='r', label=f'estimate: for passed points')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Number of data points')
        ax.set_ylabel('Runtime (seconds)')
        ax.legend()
        ax.set_title(f'{func} runtime estimate in seconds')
        plt.show()
    return out

if __name__ == '__main__':
    for f in ['MannKendall', 'SeasonalKendall', 'MultiPartKendall_2part', 'SeasonalMultiPartKendall_2part',
                        'MultiPartKendall_3part', 'SeasonalMultiPartKendall_3part']:
        print(f, estimate_runtime(np.array([500, 1000,5000,10000]), f, plot=True))
