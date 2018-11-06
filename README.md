# Piecewise Linear Time Warping

This repo contains research code for time warping multi-dimensional time series, with a focus on neural data. The core functions are stable, but still under development.

The code fits time warping models with either linear or piecewise linear warping functions. These models are more constrained than the classic [Dynamic Time Warping (DTW)](https://en.wikipedia.org/wiki/Dynamic_time_warping) algorithm, and are thus less prone to overfit to data with high levels of noise. This is demonstrated below on synthethic data. Briefly, a 1-dimensional time series is measured over many repetitions (trials), and exhibits a similar temporal profile but with random jitter on each trial ("misaligned data"). Simply averaging across trials produces a poor description of the typical time series (red trace at bottom). A linear time warping model identifies a much better prototypical trace ("template"), while accounting for the temporal translations on each trial with warping functions (blue to red linear functions at bottom). On the right, a nonlinear warping model based on DTW (called [DBA](https://github.com/fpetitjean/DBA)) is shown for comparison. While DBA can work well on datasets with low noise, linear warping models can be easier to interpret and less likely to overfit.

<img width="1445" alt="screen shot 2018-11-05 at 2 03 55 pm" src="https://user-images.githubusercontent.com/636625/48030119-e3a28d80-e104-11e8-8932-c1251f168f4b.png">

## Getting started

After installing (see [below](#installing)), check out the demos in the [`examples/`](https://github.com/ahwillia/affinewarp/tree/master/examples) folder.

* `Shift.ipynb` - demonstrates the essential ideas on a very simple, synthethic dataset containing only one neuron.
* `OFC-2.ipynb` - shows a more complete analysis on real data. The tutorial uses a publicly available dataset from [crcns.org](https://crcns.org/), which can be found [here](https://crcns.org/data-sets/ofc/ofc-2/about-ofc-2). See `OFC-2-wrangle-data.ipynb` for instructions on downloading and organizing the dataset.

The code is fairly well-documented but the tutorials can still be improved, so let us know if you run into trouble. 

## Installing

This package isn't registered yet, so you need to install manually. Either download or clone the repo:

```
git clone https://github.com/ahwillia/affinewarp/
```

Then navigate to the downloaded folder:

```
cd /path/to/affinewarp
```

And install the package:

```
pip install .
```

You will need to repeat these steps if we update the code.

## Other references / resources

* [tw-pca](https://github.com/ganguli-lab/twpca) - Time-Warping Principal Components Analysis, also supports linear and shift-only warping functions. Does not support piecewise linear warping functions and assumes that time series are low-dimensional. Nonlinear warping methods are also supported. See our [conference abstract](https://cs.stanford.edu/~poole/twpca_poster.pdf) and [poster](https://cs.stanford.edu/~poole/twpca_poster.pdf).

* [tslearn](https://tslearn.readthedocs.io/) - A Python package supporting a variety of time series models, including DTW-based methods.

## Contact

ahwillia@stanford.edu (or open an issue here).
