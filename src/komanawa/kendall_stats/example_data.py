"""
created matt_dumont 
on: 21/09/23
"""
import numpy as np
import pandas as pd
from copy import deepcopy


def make_increasing_decreasing_data(slope=1, noise=1, step=1):
    """
    make increasing and decreasing test data
    :param slope: slope for line
    :param noise: random noise to add to data
    :param step: spacing of data
    :return: x,y (np.arrays)
    """
    x = np.arange(0, 100, step).astype(float)
    y = x * slope
    np.random.seed(68)
    noise = np.random.normal(0, noise, len(x))
    y += noise
    return x, y


def make_seasonal_data(slope, noise, unsort, na_data, step=1):
    """
    make seasonal test data
    :param slope: slope for line
    :param noise: noise to add to data
    :param unsort: make the data unsorted (for testing purposes)
    :param na_data: add some na data to the data (for testing purposes)
    :param step: spacing of data
    :return: pd.DataFrame
    """
    x, y = make_increasing_decreasing_data(slope=slope, noise=noise, step=step)
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat(np.array([[1, 2, 3, 4]]), len(x) // 4+1, axis=0).flatten()
    seasons = seasons[:len(x)]
    y[seasons == 1] += 0 * noise / 2
    y[seasons == 2] += 2 * noise / 2
    y[seasons == 3] += 0 * noise / 2
    y[seasons == 4] += -2 * noise / 2

    if na_data:
        np.random.seed(868)
        na_idxs = np.random.randint(0, len(y), 10)
        y[na_idxs] = np.nan

    # test passing data col (with other noisy cols)
    test_dataframe = pd.DataFrame(index=x, data=y, columns=['y'])
    test_dataframe['seasons'] = seasons
    for col in ['lkj', 'lskdfj', 'laskdfj']:
        test_dataframe[col] = np.random.choice([1, 34.2, np.nan])
    if unsort:
        x_use = deepcopy(x)
        np.random.shuffle(x_use)
        test_dataframe = test_dataframe.loc[x_use]

    return test_dataframe


def make_multipart_sharp_change_data(slope, noise, unsort, na_data, step=1):
    """
    sharp v change positive slope is increasing and then decreasing, negative is opposite
    :param slope: slope for line
    :param noise: noise to add to data
    :param unsort: make the data unsorted (for testing purposes)
    :param na_data: add some na data to the data (for testing purposes)
    :return: x,y (np.arrays)
    """
    x = np.arange(0, 100, step).astype(float)
    y = np.zeros_like(x).astype(float)
    sp = len(x) // 2
    y[:sp] = x[:sp] * slope + 100
    y[sp:] = (x[sp:] - x[sp - 1].max()) * slope * -1 + y[sp - 1]

    np.random.seed(68)
    noise = np.random.normal(0, noise, len(x))
    y += noise

    if na_data:
        np.random.seed(868)
        na_idxs = np.random.randint(0, len(y), 10)
        y[na_idxs] = np.nan

    if unsort:
        x_use = np.arange(len(x))
        np.random.shuffle(x_use)
        y = y[x_use]
        x = x[x_use]

    return x, y


def make_multipart_parabolic_data(slope, noise, unsort, na_data, step=1):
    """
    note the slope is multiplied by -1 to retain the same standards make_sharp_change_data
    positive slope is increasing and then decreasing, negative is opposite
    :param slope: slope for line
    :param noise: noise to add to data
    :param unsort: make the data unsorted (for testing purposes)
    :param na_data: add some na data to the data (for testing purposes)
    :return: x,y (np.arrays)
    """

    x = np.arange(0, 100, step).astype(float)
    y = slope * -1 * (x - 49) ** 2 + 100.

    np.random.seed(68)
    noise = np.random.normal(0, noise, len(x))
    y += noise

    if na_data:
        np.random.seed(868)
        na_idxs = np.random.randint(0, len(y), 10)
        y[na_idxs] = np.nan

    if unsort:
        x_use = deepcopy(x)
        np.random.shuffle(x_use)
        y = y[x_use]
        x = x[x_use]

    return x, y


def make_seasonal_multipart_parabolic(slope, noise, unsort, na_data, step=1):
    """
    make seasonal test data
    :param slope: slope for parabola note the slope is multiplied by -1 to retain the same standards make_sharp_change_data
    :param noise: noise to add to data
    :param unsort: make the data unsorted (for testing purposes)
    :param na_data: add some na data to the data (for testing purposes)
    :param step: spacing of data
    :return: pd.DataFrame
    """
    x, y = make_multipart_parabolic_data(slope=slope, noise=noise, unsort=False, na_data=False, step=step)
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat(np.array([[1, 2, 3, 4]]), len(x) // 4 +1, axis=0).flatten()
    seasons = seasons[:len(x)]
    y[seasons == 1] += 0 + noise / 4
    y[seasons == 2] += 2 + noise / 4
    y[seasons == 3] += 0 + noise / 4
    y[seasons == 4] += -2 + noise / 4

    if na_data:
        np.random.seed(868)
        na_idxs = np.random.randint(0, len(y), 10)
        y[na_idxs] = np.nan

    # test passing data col (with other noisy cols)
    test_dataframe = pd.DataFrame(index=x, data=y, columns=['y'])
    test_dataframe['seasons'] = seasons
    for col in ['lkj', 'lskdfj', 'laskdfj']:
        test_dataframe[col] = np.random.choice([1, 34.2, np.nan])
    if unsort:
        x_use = deepcopy(x)
        np.random.shuffle(x_use)
        test_dataframe = test_dataframe.loc[x_use]

    return test_dataframe


def make_seasonal_multipart_sharp_change(slope, noise, unsort, na_data, step=1):
    """
    make seasonal test data
    :param slope: slope for line
    :param noise: noise to add to data
    :param unsort: make the data unsorted (for testing purposes)
    :param na_data: add some na data to the data (for testing purposes)
    :param step: spacing of data
    :return: pd.DataFrame
    """
    x, y = make_multipart_sharp_change_data(slope=slope, noise=noise, unsort=False, na_data=False, step=step)
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat(np.array([[1, 2, 3, 4]]), len(x) // 4 + 1, axis=0).flatten()
    seasons = seasons[:len(x)]
    y[seasons == 1] += 0 + noise / 4
    y[seasons == 2] += 2 + noise / 4
    y[seasons == 3] += 0 + noise / 4
    y[seasons == 4] += -2 + noise / 4

    if na_data:
        np.random.seed(868)
        na_idxs = np.random.randint(0, len(y), 10)
        y[na_idxs] = np.nan

    # test passing data col (with other noisy cols)
    test_dataframe = pd.DataFrame(index=x, data=y, columns=['y'])
    test_dataframe['seasons'] = seasons
    for col in ['lkj', 'lskdfj', 'laskdfj']:
        test_dataframe[col] = np.random.choice([1, 34.2, np.nan])
    if unsort:
        x_use = deepcopy(x)
        np.random.shuffle(x_use)
        test_dataframe = test_dataframe.loc[x_use]

    return test_dataframe


multipart_sharp_slopes = [0.1, -0.1, 0]
multipart_sharp_noises = [0, 0.5, 1, 5]
slope_mod = 1e-2
multipart_parabolic_slopes = [1 * slope_mod, -1 * slope_mod, 0]
multipart_parabolic_noises = [0, 1, 5, 10, 20, 50]
