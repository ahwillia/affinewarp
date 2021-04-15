# Piecewise Linear Time Warping

This repo contains research code for time warping multi-dimensional time series. This was developed as part of the following manuscript, which focuses on analysis of large-scale neural recordings (though this code can be also be applied to many other data types):

> [Discovering precise temporal patterns in large-scale neural recordings through robust and interpretable time warping](https://doi.org/10.1016/j.neuron.2019.10.020).<br>
Williams AH, Poole B, Maheswaranathan N, Dhawale AK, Fisher T, Wilson CD, Brann DH, Trautmann E, Ryu S, Shusterman R, Rinberg D, Ölveczky BP, Shenoy KV, Ganguli S (2020). *Neuron*. 105(2):246-259.e8

The code fits time warping models with either linear or piecewise linear warping functions. These models are more constrained than the classic [Dynamic Time Warping (DTW)](https://en.wikipedia.org/wiki/Dynamic_time_warping) algorithm, and are thus less prone to overfit to data with high levels of noise. This is demonstrated below on synthethic data. Briefly, a 1-dimensional time series is measured over many repetitions (trials), and exhibits a similar temporal profile but with random jitter on each trial. Simply averaging across trials produces a poor description of the typical time series (red trace at bottom). A linear time warping model identifies a much better prototypical trace (labeled "template"), while accounting for the temporal translations on each trial with warping functions (blue to red linear functions at bottom). On the right, a nonlinear warping model based on DTW (called [DBA](https://github.com/fpetitjean/DBA)) is shown for comparison. While DBA can work well on datasets with low noise, linear warping models can be easier to interpret and less likely to overfit.

<img width="1445" alt="screen shot 2018-11-05 at 2 03 55 pm" src="https://user-images.githubusercontent.com/636625/48030119-e3a28d80-e104-11e8-8932-c1251f168f4b.png">

## Getting started

After installing (see below), check out the demos in the [`examples/`](https://github.com/ahwillia/affinewarp/tree/master/examples) folder.

Either download or clone the repo:

```
git clone https://github.com/ahwillia/affinewarp/
```

Then navigate to the downloaded folder:

```
cd /path/to/affinewarp
```

Install the package and requirements:

```
pip install .
pip install -r requirements.txt
```

You will need to repeat these steps if we update the code.

## Other references / resources

* [tslearn](https://tslearn.readthedocs.io/) - A Python package supporting a variety of time series models, including DTW-based methods.

## Contact

ahwillia@stanford.edu (or open an issue here).
