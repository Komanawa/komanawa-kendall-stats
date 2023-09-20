"""
created matt_dumont 
on: 21/09/23
"""
import  numpy as np
import pandas as pd
from copy import deepcopy

def make_increasing_decreasing_data(slope=1, noise=1):
    x = np.arange(100).astype(float)
    y = x * slope
    np.random.seed(68)
    noise = np.random.normal(0, noise, len(x))
    y += noise
    return x, y


def make_seasonal_data(slope, noise, unsort, na_data):
    x, y = make_increasing_decreasing_data(slope=slope, noise=noise)
    assert len(x) % 4 == 0
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat(np.array([[1, 2, 3, 4]]), len(x) // 4, axis=0).flatten()
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


def make_multipart_sharp_change_data(slope, noise, unsort, na_data):
    """
    sharp v change positive slope is increasing and then decreasing, negative is opposite
    :param slope:
    :param noise:
    :param unsort:
    :param na_data:
    :return:
    """
    x = np.arange(100)
    y = np.zeros_like(x).astype(float)
    y[:50] = x[:50] * slope + 100
    y[50:] = (x[50:] - x[49].max()) * slope * -1 + y[49]

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


def make_multipart_parabolic_data(slope, noise, unsort, na_data):
    """
    note the slope is multiplied by -1 to retain the same standards make_sharp_change_data
    positive slope is increasing and then decreasing, negative is opposite
    :param slope:
    :param noise:
    :param unsort:
    :param na_data:
    :return:
    """

    x = np.arange(100)
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


def make_seasonal_multipart_parabolic(slope, noise, unsort, na_data):
    x, y = make_multipart_parabolic_data(slope=slope, noise=noise, unsort=False, na_data=False)
    assert len(x) % 4 == 0
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat(np.array([[1, 2, 3, 4]]), len(x) // 4, axis=0).flatten()
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


def make_seasonal_multipart_sharp_change(slope, noise, unsort, na_data):
    x, y = make_multipart_sharp_change_data(slope=slope, noise=noise, unsort=False, na_data=False)
    assert len(x) % 4 == 0
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat(np.array([[1, 2, 3, 4]]), len(x) // 4, axis=0).flatten()
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

