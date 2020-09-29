"""Import commonly used libraries"""

import time
import logging
import glob
import os
import numpy as np
import pandas as pd
import collections
# from natsort import natsorted

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42 # editable text in matplotlib
mpl.rcParams['svg.fonttype'] = 'none'

import matplotlib.ticker as mtick
PercentFormat = mtick.FuncFormatter(lambda y, _: '{:.3%}'.format(y))
ScalarFormat = mtick.ScalarFormatter()

# seaborn
import seaborn as sns
sns.set_style('ticks', rc={'axes.grid':True})
sns.set_context('talk')

# set matplotlib formats
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# data structures
GC_matrix = collections.namedtuple('GC_matrix', ['gene', 'cell', 'data'])
