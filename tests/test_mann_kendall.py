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



class TestMannKendall(unittest.TestCase):



if __name__ == '__main__':
    unittest.main()