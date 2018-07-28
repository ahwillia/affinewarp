import numpy as np
from affinewarp.spikedata import SpikeData

neurons = [0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 2, 2]
trials = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
timepoints = np.random.rand(len(trials))

data = SpikeData(trials, timepoints, neurons)
