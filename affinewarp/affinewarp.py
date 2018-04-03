import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import trange, tqdm

# TODO :
# Fit only with shifts

class AffineWarping(object):
    """Represents a collection of time series, each with an affine time warp.
    """
    def __init__(self, data, max_shift=.2, max_scale=1.5):
        """
        Params
        ------
        data (ndarray) : n_trials x n_timepoints x n_features
        """

        # data dimensions
        self.data = data
        self.n_trials = data.shape[0]
        self.n_timepoints = data.shape[1]
        self.n_features = data.shape[2]

        # model options
        self.max_shift = max_shift
        self.max_scale = max_scale

        # trial-average under affine warping (initialize to random trial)
        self.template = data[np.random.randint(0, self.n_trials)].copy()
        self.tref = np.linspace(0, 1, self.n_timepoints)
        self.dt = self.tref[1]-self.tref[0]
        self.apply_warp = interp1d(self.tref, self.template, axis=0)

        # initialize shifts (taus) and scales (betas) of warping functions
        self.taus = self._sample_taus()
        self.betas = self._sample_betas()
        self.warping_funcs = self._compute_warping_funcs()

        # reconstructed data
        self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
        self.resids = self.reconstruction - self.data

        # initial loss for each trial calculation
        self.losses = np.sum(self.resids**2, axis=(1,2))
        self.loss_hist = [np.mean(self.losses)]

    def _compute_warping_funcs(self, taus=None, betas=None):
        if taus is None:
            taus = self.taus # shifts
        if betas is None:
            betas = self.betas # scales

        return np.array([np.clip(self.tref*b-t, 0, 1) for b, t in zip(betas, taus)])


    def _sample_taus(self):
        """Randomly sample shifts
        """
        return np.random.uniform(-self.max_shift, self.max_shift, size=self.n_trials)

    def _sample_betas(self):
        """Randomly sample slopes
        """
        return np.exp(np.random.uniform(np.log(1/self.max_scale), np.log(self.max_scale), size=self.n_trials))

    def randfit_warps(self, iterations=20):
        for i in trange(iterations):
            # randomly sample warping functions
            taus = self._sample_taus()
            betas = self._sample_betas()
            warps = self._compute_warping_funcs(taus, betas)

            # warp data and compute new losses
            recon = np.array([self.apply_warp(t) for t in warps])
            losses = np.sum((recon - self.data)**2, axis=(1,2))

            # update warping parameters for trials with improved loss
            idx = losses < self.losses
            self.losses[idx] = losses[idx]
            self.taus[idx] = taus[idx]
            self.betas[idx] = betas[idx]
            self.warping_funcs[idx] = warps[idx]
            self.reconstruction[idx] = recon[idx]
            self.loss_hist.append(np.mean(self.losses))

    def optimize_warps(self, maxiter=1000):
        """Fit warps by gradient-based optimization
        """

        def f(x):
            N = len(x) // 2
            wfuncs = self._compute_warping_funcs(taus=x[:N], betas=x[N:])
            recon = np.array([self.apply_warp(z) for z in wfuncs])
            resids = recon - self.data
            Lam, Z0 = np.modf(wfuncs*(self.n_timepoints-1))
            Z0 = Z0.astype(int)
            Z0[Lam < 1e-5] = -1
            Z0[Lam > (1-1e-5)] = -1

            grad = np.zeros(N*2)
            dtemp = np.zeros(self.template.shape)
            dtemp[:self.n_timepoints-1] = np.diff(self.template, axis=0) / self.dt
            for k, z0 in zip(range(N), Z0):
                grad[k] = -np.sum(resids[k]*dtemp[z0])
                grad[k+N] = np.sum(resids[k]*(dtemp[z0]*self.tref[:,None]))
            return .5*np.sum(resids**2), grad

        x0 = np.concatenate((self.taus, self.betas), axis=0)
        bounds = [(-self.max_shift, self.max_shift) for _ in range(self.n_trials)] + [(1/self.max_scale, self.max_scale) for _ in range(self.n_trials)]
        result = minimize(f, x0, method='L-BFGS-B', jac=True, bounds=bounds, options=dict(maxiter=100))
        print(result.message)
        self.taus = result.x[:self.n_trials]
        self.betas = result.x[self.n_trials:]
        self.warping_funcs = self._compute_warping_funcs()
        self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
        self.resids = self.reconstruction - self.data
        self.losses = np.sum(self.resids**2, axis=(1,2))
        self.loss_hist.append(np.mean(self.losses))

    def fit_template(self):
        # compute normal equations
        T = self.n_timepoints
        WtW = np.zeros((T+1, T+1))
        WtX = np.zeros((T+1, self.n_features))
        for wfunc, Xk in zip(self.warping_funcs, self.data):
            lam, i = np.modf(wfunc * (T-1))
            i = i.astype(int)
            WtW[i, i] += 1-lam
            WtW[i+1, i+1] += lam
            WtX[i] += lam[:,None] * Xk
            WtX[i+1] += (1-lam)[:,None] * Xk
        WtW = WtW[:T, :T]
        WtX = WtX[:T]

        # update template
        self.template = np.linalg.solve(WtW, WtX)
        self.apply_warp = interp1d(self.tref, self.template, axis=0)
    
        # update loss
        self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
        self.resids = self.reconstruction - self.data
        self.losses = np.sum(self.resids**2, axis=(1,2))
        self.loss_hist.append(np.mean(self.losses))

        return self.template


    # def fit_template2(self):

    #     Xw = self.transform(fill_value=np.nan)
    #     self.template = np.nanmean(Xw, axis=0)
    #     self.apply_warp = interp1d(self.tref, self.template, axis=0)
        
    #     # update loss
    #     self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
    #     self.resids = self.reconstruction - self.data
    #     self.losses = np.sum(self.resids**2, axis=(1,2))
    #     self.loss_hist.append(np.mean(self.losses))

    #     return self.template

        # # floored and ceiled warping functions
        # Lam, Z0 = np.modf(self.warping_funcs*(self.n_timepoints-1))
        # Lam = Lam[:,:,None]
        # Z0 = Z0.astype(int)
        # Z1 = np.clip(Z0+1, 0, self.n_timepoints-1)

        # # compute objective and derivative of template
        # def f(x):
        #     template = x.reshape((self.n_timepoints, self.n_features))
        #     g = interp1d(self.tref, template, axis=0)
        #     resids = np.array([g(w) for w in self.warping_funcs]) - self.data
        #     grad = np.zeros((self.n_timepoints, self.n_features))
        #     for res, z0, z1, lam in zip(resids, Z0, Z1, Lam):
        #         grad[z0] += (1-lam)*res
        #         grad[z1] += lam*res
        #     return .5*np.sum(resids**2), grad.ravel()
       
        # result = minimize(f, self.template.ravel(), method='L-BFGS-B', jac=True)
        # print(result.message)
        # print('# of iterations: {}'.format(result.nit))
        # self.template = np.reshape(result.x, self.template.shape)
        # self.apply_warp = interp1d(self.tref, self.template, axis=0)

        # # update loss
        # self.reconstruction = np.array([self.apply_warp(t) for t in self.warping_funcs])
        # self.resids = self.reconstruction - self.data
        # self.losses = np.sum(self.resids**2, axis=(1,2))
        # self.loss_hist.append(np.mean(self.losses))

        # return self.template

    def transform(self, data=None, fill_value=0):
        data = self.data if data is None else data
        warped_data = np.empty(data.shape)
        for k in range(self.n_trials):
            self.warping_funcs[k]
            # f = interp1d(self.warping_funcs[k], self.tref, axis=0, bounds_error=False, fill_value='extrapolate')
            # g = interp1d(self.tref, data[k], axis=0, bounds_error=False, fill_value=fill_value)
            # warped_data[k] = g(f(self.tref))
        return warped_data
