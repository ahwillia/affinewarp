# Piecewise Linear Time Warping

This repo contains research code for time warping multi-dimensional time series, with a focus on neural data. The core functions are stable, but still under development.

The code fits time warping models with either linear or piecewise linear warping functions. These models are more constrained than the classic [Dynamic Time Warping (DTW)](https://en.wikipedia.org/wiki/Dynamic_time_warping) algorithm, and are thus less prone to overfit to data with high levels of noise. This is demonstrated below on synthethic data. Briefly, a 1-dimensional time series is measured over many repetitions (trials), and exhibits a similar temporal profile but with random jitter on each trial ("misaligned data"). Simply averaging across trials produces a poor description of the typical time series (red trace at bottom). A linear time warping model identifies a much better prototypical trace ("template"), while accounting for the temporal translations on each trial with warping functions (blue to red linear functions at bottom). On the right, a nonlinear warping model based on DTW (called [DBA](https://github.com/fpetitjean/DBA)) is shown for comparison. While DBA can work well on datasets with low noise, linear warping models can be easier to interpret and less likely to overfit.

<img width="1445" alt="screen shot 2018-11-05 at 2 03 55 pm" src="https://user-images.githubusercontent.com/636625/48030119-e3a28d80-e104-11e8-8932-c1251f168f4b.png">

## Getting started

After installing (see below). See the jupyter notebook demos in the [`examples/`](https://github.com/ahwillia/affinewarp/tree/master/examples) folder. The code is fairly well-documented but the tutorials can still be improved, so open issues if you run into trouble.

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

