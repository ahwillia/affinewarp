
class FitHistory(object):

    """docstring for Population"""
    def __init__(self, param_dims, popsize=100, paramtype=np.float64):

        if not np.iterable(param_dims):
            param_dims = (param_dims,)

        self.popsize = popsize
        self.losses = np.empty(popsize)
        self.params = np.empty((popsize, *param_dims))
        self._n = 0

    def insert(self, new_param, new_loss):

        # if we aren't at capacity add new member
        if len(self.H) < self.popsize:
            heappush(self.H, (-new_loss, new_param))

        # if we are at capacity, only inset if new member has lower loss
        else:
            if new_loss self.H[0][0]

        if self._n < (self.popsize - 1):

            self.params[self._n] = new_param
            self._n += 1

        # keep new parameters if new loss is lower than worst in population
        elif new_loss < self.losses[-1]:
