Komanawa Kendall Stats
#########################################

a small repo that holds:

* MannKendall: non parametric trend detection
* SeasonalKendall: non parametric trend detection with seasonality
* MultiPartKendall: non parametric change point detection
* SeasonalMultiPartKendall: non parametric change point detection with seasonality

.. toctree::
    :maxdepth: 2
    :hidden:

    Code documentation<autoapi/komanawa/kendall_stats/index.rst>

Dependencies
=============

* pandas>=2.0.3
* numpy>=1.25.2
* matplotlib>=3.7.2
* scipy>=1.11.2
* pytables>=3.8.0

The following command will install a virtual environment with all the dependencies:


.. code-block:: bash

    conda create -c conda-forge --name kendall python=3.11 pandas=2.0.3 numpy=1.25.2 matplotlib=3.7.2 scipy=1.11.2 pytables=3.8.0

Install
========

Install from pypi
-----------------

.. todo

This is coming soon

Install from github
--------------------

we manage this package as a simple github package, but it can still be installed by pip:

For the latest version use:

.. code-block:: bash

    pip install git+https://github.com/Komanawa-Solutions-Ltd/komanawa-kendall-stats.git

For a specific version use:

.. code-block:: bash

    pip install git+https://github.com/Komanawa-Solutions-Ltd/komanawa-kendall-stats.git@{version}

    # example:
    pip install git+https://github.com/Komanawa-Solutions-Ltd/komanawa-kendall-stats.git@v1.0.0

Update
=======

 option 1 is to uninstall the package with pip then reinstall it.

.. code-block:: bash

    pip uninstall komanawa-kendall-stats

otherwise, use pip update option

.. code-block:: bash

    pip install git+https://github.com/Komanawa-Solutions-Ltd/komanawa-kendall-stats.git -U


Usage
=======

Full documentation can be found in the docstrings, but the basic usage is provided below

MannKendall
-----------

.. code-block:: python

    import matplotlib.pyplot as plt
    import pandas as pd
    from komanawa.kendall_stats import MannKendall, example_data

    x, y = example_data.make_increasing_decreasing_data(slope=0.1, noise=5)

    # to pass a simple array (no index) use data_col=None, assumes regular sampling
    mk = MannKendall(data=y, alpha=0.05, data_col=None, rm_na=True)

    # to assume irregular sampling, pass a pandas series, with a sortable index
    mk = MannKendall(data=pd.Series(y, index=x), alpha=0.05, data_col=None, rm_na=True)

    # to pass a dataframe, pass a data_col, and a dataframe with a sortable index
    mk = MannKendall(data=pd.DataFrame(index=x, data=y, columns=['y']), alpha=0.05, data_col='y', rm_na=True)

    # note that by default, nan values are removed and the data is sorted via the series/Dataframe index,
    # where no index is passed (e.g., np.ndarray) the index is assumed to be np.arange(len(data))

    # the trend is accessed via the trend attribute,
    print(mk.trend)

    # note the trend is stored as int (1, 0, -1) for increasing, no trend, decreasing
    # to convert to a string use the trend_dict attribute
    print(mk.trend_dict[mk.trend])

    # other attributes
    print(mk.p)  # p value
    print(mk.z) # z value

    # there are two convenience methods

    # calculate the senslope of the data
    print(mk.calc_senslope())

    # plot the data and the trend
    fig, ax = mk.plot_data()
    ax.set_title('Example Mann Kendall')
    plt.show()


.. figure:: ../figures/example_mk.png
   :height: 500 px
   :align: center


SeasonalKendall
----------------
SeasonalKendall is as per MannKendall, but with a seasonal component.


.. code-block:: python

    import matplotlib.pyplot as plt
    import pandas as pd
    from komanawa.kendall_stats import SeasonalKendall, example_data

    data = example_data.make_seasonal_data(slope=0.1, noise=5, unsort=False, na_data=False)
    assert isinstance(data, pd.DataFrame)
    print(data)

    # you must pass a dataframe with at least a column of data and a column of seasons for the seasonal kendall
    smk = SeasonalKendall(df=data, alpha=0.05, data_col='y', season_col='seasons', rm_na=True)

    # otherwise the SeasonalKendall class is the same as the MannKendall class
    # note that by default, nan values are removed and the data is sorted via the series/Dataframe index,
    # where no index is passed (e.g., np.ndarray) the index is assumed to be np.arange(len(data))

    print(smk.trend) # trend as int
    print(smk.trend_dict[smk.trend]) # trend as string
    print(smk.p)  # p value
    print(smk.z) # z value

    # calculate the senslope of the data
    print(smk.calc_senslope())

    # plot the data and the trend
    fig, ax = smk.plot_data()
    ax.set_title('example seasonal kendall')
    plt.show()

.. figure:: ../figures/example_smk.png
   :height: 500 px
   :align: center

MultiPartKendall
-----------------

.. code-block:: python

    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd
    from komanawa.kendall_stats import MultiPartKendall, example_data
    plot_dir = Path.home().joinpath('Downloads', 'mk_plots')
    plot_dir.mkdir(exist_ok=True)

    x_sharp, y_sharp = example_data.make_multipart_sharp_change_data(slope=example_data.multipart_sharp_slopes[0],
                                                                     noise=example_data.multipart_sharp_noises[2],
                                                                     unsort=False,
                                                                     na_data=False)
    data = pd.Series(index=x_sharp, data=y_sharp)
    serial_path = Path.home().joinpath('Downloads', 'multipart_mk.hdf')
    serial_path2 = Path.home().joinpath('Downloads', 'multipart_mk2.hdf')
    mk = MultiPartKendall(
        data=data,  # data can be passed as a np.array, pd.Series, or pd.DataFrame
        nparts=2,  # number of parts to split data into
        expect_part=(1, -1),  # the expected slope of each part (1, increasing, 0, no change, -1, decreasing)
        min_size=10,
        data_col=None,
        alpha=0.05,  # significance level for trends (p<alpha)
        no_trend_alpha=0.5,  # significance level for no trend (p>no_trend_alpha)
        rm_na=True,
        serialise_path=serial_path,  # None or path to serialise results to
        recalc=False)

    # the serialised results can be loaded back in by simply re-running the constructor with the same serialise_path
    # if recalc is False (default) the results will be loaded from the serialised file, otherwise they will be recalculated
    # and re-saved to the serialised file e.g.
    mk1 = MultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10, data_col=None, alpha=0.05,
                           no_trend_alpha=0.5,
                           rm_na=True, serialise_path=serial_path, recalc=False)

    # equivalency is managed and will test all inputs, but the loaded object will have a different id
    assert mk == mk1
    assert id(mk) != id(mk1)

    # you can also create an instance from a file
    mk2 = MultiPartKendall.from_file(serial_path)

    # you can also explicitly save the results to a file
    mk.to_file(save_path=serial_path2, complevel=9, complib='blosc:lz4')

    # the class calculates the kendal slope for all data subsets
    # (e.g. for the above slopes will be calculated for the data split at
    # n where n >= min_size and n<= len(data)-min_size)

    # the user can get all acceptable matches where:
    # * the pvalue meets the criterion (p<alpha for data with trends, p>no_trend_alpha for data with no trend)
    # * the trend of the data matches the expected trend
    mk.get_acceptable_matches()

    # the user can get all matches via
    mk.get_all_matches()

    # the user can plot match statistics ('p', 'z', 's', 'var_s') for all matches
    fig, ax = mk.plot_acceptable_matches('z')
    ax.set_title('z statistic for all acceptable matches')
    fig.savefig(plot_dir.joinpath('multi_mk_z.png'))
    plt.show()




.. figure:: ../figures/multi_mk_z.png
   :height: 500 px
   :align: center



.. code-block:: python

    # the user can get the best match(s)
    # the best match is the maximum znorm_joint value where:
    #   if expected trend == 1 or -1:
    #     znorm = the min-max normalised z value for each part
    #   else: (no trend expected)
    #     znorm = 1 - the min-max normalised z value for each part
    #   and
    #     znorm_joint = the sum of the znorm values for each part
    # when there are multiple matches with the same znorm_joint value both will be returned
    # when there are no acceptable matches None will be returned
    # or a ValueError will be raised if raise_on_none=True
    best_points = mk.get_maxz_breakpoints()


    # the user can get the data, and kendall stats for a specific breakpoint
    data, kendall_stats = mk.get_data_from_breakpoints(breakpoints=50) # get the data split at point 50
    print(data) # list containing the data for each part
    # kendal stats for each part: 'trend', 'h', 'p', 'z', 's', 'var_s', 'senslope', 'senintercept'
    print(kendall_stats)

    # the user can also plot the data from a specific breakpoint
    fig, ax = mk.plot_data_from_breakpoints(breakpoints=50, txt_vloc=-0.05, add_labels=True)
    ax.set_title('data split at 50')
    fig.savefig(plot_dir.joinpath('multi_mk_data.png'))
    plt.show()

.. figure:: ../figures/multi_mk_data.png
   :height: 500 px
   :align: center

.. code-block:: python

    # user can specify any number of parts to split data into
    x_para, y_para = example_data.make_multipart_parabolic_data(slope=example_data.multipart_parabolic_slopes[0],
                                                                noise=example_data.multipart_parabolic_noises[2],
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
    fig, ax = mk.plot_data_from_breakpoints(breakpoints=[40, 60], txt_vloc=-0.05, add_labels=True)
    ax.set_title('data split at 40 and 60')
    fig.savefig(plot_dir.joinpath('multi_mk_data2.png'))
    plt.show()

.. figure:: ../figures/multi_mk_data2.png
   :height: 500 px
   :align: center

SeasonalMultiPartKendall
-------------------------
The SeasonalMultiPartKendall is as per the MultiPartKendall, but with a seasonal component.

.. code-block:: python

    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd
    from komanawa.kendall_stats import SeasonalMultiPartKendall, example_data
    plot_dir = Path.home().joinpath('Downloads', 'smk_plots')
    plot_dir.mkdir(exist_ok=True)

    data = example_data.make_seasonal_multipart_sharp_change(slope=example_data.multipart_sharp_slopes[0],
                                                                     noise=example_data.multipart_sharp_noises[2],
                                                                     unsort=False,
                                                                     na_data=False)
    # initalisation is identical to MultiPartKendall except that data must be a DataFrame
    # and data_col and seasonal_col must be specified

    smk = SeasonalMultiPartKendall(
        data=data,  # data can be passed as a np.array, pd.Series, or pd.DataFrame
        nparts=2,  # number of parts to split data into
        expect_part=(1, -1),  # the expected slope of each part (1, increasing, 0, no change, -1, decreasing)
        min_size=10,
        data_col='y',
        season_col='seasons',
        alpha=0.05,  # significance level for trends (p<alpha)
        no_trend_alpha=0.5,  # significance level for no trend (p>no_trend_alpha)
        rm_na=True,
        serialise_path=None,  # None or path to serialise results to
        recalc=False)

    # the user can also plot the data from a specific breakpoint
    fig, ax = smk.plot_data_from_breakpoints(breakpoints=50, txt_vloc=-0.05, add_labels=True)
    ax.set_title('data split at 50')
    fig.savefig(plot_dir.joinpath('multi_smk_data.png'))
    plt.show()

.. figure:: ../figures/multi_smk_data.png
   :height: 500 px
   :align: center

.. code-block:: python

    # user can specify any number of parts to split data into
    data = example_data.make_seasonal_multipart_parabolic(slope=example_data.multipart_parabolic_slopes[0],
                                                                noise=example_data.multipart_parabolic_noises[2],
                                                                unsort=False,
                                                                na_data=False)
    smk = SeasonalMultiPartKendall(
        data=data,  # data can be passed as a np.array, pd.Series, or pd.DataFrame
        nparts=3,  # number of parts to split data into
        expect_part=(1, 0, -1),  # the expected slope of each part (1, increasing, 0, no change, -1, decreasing)
        min_size=10,
        data_col='y',
        season_col='seasons',
        alpha=0.05,  # significance level for trends (p<alpha)
        no_trend_alpha=0.5,  # significance level for no trend (p>no_trend_alpha)
        rm_na=True,
        serialise_path=None,  # None or path to serialise results to
        recalc=False)
    fig, ax = smk.plot_data_from_breakpoints(breakpoints=[40, 60], txt_vloc=-0.05, add_labels=True)
    ax.set_title('data split at 40 and 60')
    fig.savefig(plot_dir.joinpath('multi_smk_data2.png'))
    plt.show()

.. figure:: ../figures/multi_smk_data2.png
    :height: 500 px
    :align: center

Resource Requirements
=======================

Note that Mann Kendall tests require calculating the covariance matrix of the data, which creates a NxN matrix where N is the length of the data.  In addition the Multipart Kendall analysis requires indexing all possible combinations of data splits, which is a factorial problem.  As such, the resource requirements for these tests can be quite high.  The following table provides some rough estimates of the resource requirements for the tests.

Theoretical memory requirements
--------------------------------

* N: 4 * s_array memory
* 50: 8e-05 gb
* 100: 0.00032 gb
* 500: 0.008 gb
* 1,000: 0.032 gb
* 5,000: 0.8 gb
* 10,000: 3.2 gb
* 25,000: 20.0 gb
* 50,000: 80.0 gb

Processing Time
-----------------

The table of processing times was run on a single thread (11th Gen Intel(R) Core(TM) i5-11500H @ 2.90GHz with 32 GB of DDR4 RAM). The results are in seconds If you want a processing time table for a different machine run:

.. code-block:: python

    from pathlib import Path
    from komanawa.kendall_stats.time_tests import run_time_test
    run_time_test(outdir=Path.home().joinpath('Downloads'))

Note that this may take some time

+---------+-----------------------+---------------------------+----------------------------------+------------------------------------------+----------------------------------+------------------------------------------+
| npoints | MannKendall_time_test | SeasonalKendall_time_test | MultiPartKendall_2part_time_test | SeasonalMultiPartKendall_2part_time_test | MultiPartKendall_3part_time_test | SeasonalMultiPartKendall_3part_time_test |
+=========+=======================+===========================+==================================+==========================================+==================================+==========================================+
| 50      | 1.13E-03              | 4.21E-03                  | 1.65E-02                         | 2.15E-02                                 | 5.19E-02                         | 7.75E-02                                 |
+---------+-----------------------+---------------------------+----------------------------------+------------------------------------------+----------------------------------+------------------------------------------+
| 100     | 5.10E-04              | 1.76E-03                  | 2.91E-02                         | 6.25E-02                                 | 4.18E-01                         | 7.62E-01                                 |
+---------+-----------------------+---------------------------+----------------------------------+------------------------------------------+----------------------------------+------------------------------------------+
| 500     | 3.04E-03              | 5.89E-03                  | 1.93E-01                         | 7.60E-01                                 | 2.04E+01                         | 6.48E+01                                 |
+---------+-----------------------+---------------------------+----------------------------------+------------------------------------------+----------------------------------+------------------------------------------+
| 1000    | 9.40E-03              | 3.36E-02                  | 7.65E-01                         | 6.40E+00                                 | 1.27E+02                         | 6.91E+02                                 |
+---------+-----------------------+---------------------------+----------------------------------+------------------------------------------+----------------------------------+------------------------------------------+
| 5000    | 2.08E+00              | 3.04E+00                  | 6.33E+01                         | 3.97E+03                                 | na                               | na                                       |
+---------+-----------------------+---------------------------+----------------------------------+------------------------------------------+----------------------------------+------------------------------------------+
| 10000   | 9.81E+00              | 1.50E+01                  | 4.03E+02                         | 4.18E+04                                 | na                               | na                                       |
+---------+-----------------------+---------------------------+----------------------------------+------------------------------------------+----------------------------------+------------------------------------------+

Options for reduced computational requirements (MultipartMannKendall and MultipartSeasonalMannKendall)
---------------------------------------------------------------------------------------------------------

The MultipartMannKendall and MultipartSeasonalMannKendall classes have a two options to reduce the computational requirements of the analysis.  These options are:

1. changing the check_step parameter, which controls the step size of the data splits (e.g. when check_step=1, all possible data splits are checked, when check_step=2, every second data split is checked, etc.)
2. changing the check_window parameter, which controls the data which will be checked for matches.  For example when check_window=None all data splits will be checked, when check_window=(30,60) for a two part kendall only data splits between 30 and 60 will be checked.  Similarly, when check_window=[(30,50), (60,80) for a three part kendall only data splits between 30 and 50, and 60 and 80 will be checked for the first and and second breakpoints, respectively.

Either of these options can significantly reduce the runtime of the analysis, but may influance the choice of breakpoint, so use these options with care.  The table below demonstrates the time gains from these approaches on a MultipartMannKendall with 1000 datapoints. Results are in seconds.  Note that the full s_array (nxn) is still calculated, so these approaches will not reduce the memory requirements of the analysis.

+----------+--------------+--------------------+--------------------+-------------------+-------------------+
|          | window: None | window: (300, 700) | window: (400, 600) | window: (450,550) | window: (475,525) |
+==========+==============+====================+====================+===================+===================+
| step: 1  | 0.68         | 0.27               | 0.18               | 0.08              | 0.10              |
+----------+--------------+--------------------+--------------------+-------------------+-------------------+
| step: 2  | 0.35         | 0.18               | 0.08               | 0.08              | 0.04              |
+----------+--------------+--------------------+--------------------+-------------------+-------------------+
| step: 5  | 0.20         | 0.07               | 0.08               | 0.03              | 0.07              |
+----------+--------------+--------------------+--------------------+-------------------+-------------------+
| step: 10 | 0.10         | 0.09               | 0.03               | 0.07              | 0.03              |
+----------+--------------+--------------------+--------------------+-------------------+-------------------+
