"""affinewarp - Time warping under affine warping functions"""

__version__ = '0.1.0'
__author__ = 'Alex Williams <ahwillia@stanford.edu>'

from .piecewisewarp import PiecewiseWarping
from .shiftwarp import ShiftWarping
from .spikedata import SpikeData

# Experimental code.
#
# from .multiwarp import MultiShiftWarping
# from .glomwarp import AgglomerativeWarping, StagewiseWarping
