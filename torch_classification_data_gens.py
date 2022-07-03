import lab as B
import torch
import random
import numpy as np
from statistics import median


class example_data_gen:

    def __init__(self, means, covariances, num_batches = 16, dim_x: int = None, nc_bounds: list = [10, 20], nt_bounds: list = [10, 20], priors: list = [0.5, 0.5], device='cpu'):
        '''
        Returns a single epoch of batches (tasks) consisting of context and target set from a binary Mixture of Gaussians distribution with specified prior.

        Args:
        ----------
        means: List (of length 2 for benoulli) of the means of the mixture components. Each element of this list must have length equal to dim_x (if specified).
        covariances: List (of length 2 for benoulli) of the covariances of the mixture components. Each element must have shape (dim_x, dim_x) (if specified).
        num_batches: Number of batches to produce in an epoch.
        dim_x: (Optional, defaults to None) The dimensions of the domain. If 'None', it is deduced from means shape.
        nc_bounds: (Optional, defaults to [10, 20]) Bounds of the number of context points per batch. 
            The number is sampled uniformally between these. Of type [lower (int), upper (int)].
        nt_bounds: (Optional, defaults to [10, 20]) Bounds of the number of target points per batch. 
            The number is sampled uniformally between these. Of type [lower (int), upper (int)].
        split: (Optional, defaults to 0.5) The split of class 1 to class 0 data (equal to the prior probabilities of classes). A.K.A. P(y = 1).

        Returns:
        ----------
        xc: Context set inputs.
        yc: Context set outputs.
        xt: Target set inputs.
        yt: Target set outputs.
        '''

        self.num_batches = num_batches
        self.means = means
        self.covariances = covariances
        if not dim_x:
            try:
                dim_x = len(means[0])
            except:
                dim_x = 1
        self.dim_x = dim_x
        self.nc_bounds = nc_bounds
        self.nt_bounds = nt_bounds
        self.priors = priors
        self.num_classes = len(priors)
        self.device = device


    def epoch(self):

        with B.on_device(self.device):

            ##NOTE: add noise to means and covariances so that they are different tasks for meta learning:
            ##NOTE: new covariances must be PSD!
            _means = []
            _covars = []
            for i, mean in enumerate(self.means):
                _mean = torch.tensor(mean).clone().to(self.device)
                _covar = torch.tensor(self.covariances[i]).clone().to(self.device)
                _means.append(_mean + B.random.randn(torch.float32, *_mean.shape) * B.max(B.abs(_mean)))
                _cov_perturbation = B.random.randn(torch.float32, *_covar.shape)
                _covars.append(_covar + 0.2 * _cov_perturbation * B.transpose(_cov_perturbation)) ## make PSD for all random matrices

            def convert_data(points):
                _xc, _yc, _xt, _yt = points
                xc = B.transpose(_xc) ## B.transpose defaults to switching the last 2 axes: (n, c) => (c, n)
                xt = B.transpose(_xt)
                yc = B.cast(torch.float32, B.expand_dims(_yc, axis=0))
                yt = B.cast(torch.float32, B.expand_dims(_yt, axis=0))
                return xc, yc, xt, yt

            epoch = []        
            for _ in range(self.num_batches):
                _points = self.batch(_means, _covars, self.dim_x, self.nc_bounds, self.nt_bounds, self.priors)
                xc, yc, xt, yt = convert_data(_points)

                batch = {
                    'xc': xc.to(self.device),
                    'yc': yc.to(self.device),
                    'xt': xt.to(self.device),
                    'yt': yt.to(self.device),
                }
                epoch.append(batch)

            return epoch   


    def batch(self, means, covariances, dim_x, nc_bounds, nt_bounds, priors):
        
        nc = B.squeeze(torch.randint(low=nc_bounds[0], high=nc_bounds[1]+1, size=(1, )))
        nt = B.squeeze(torch.randint(low=nt_bounds[0], high=nt_bounds[1]+1, size=(1,)))
        num_points = nc + nt

        ys = []
        for i in range(self.num_classes):
            ys.extend([i] * int(num_points * priors[i]))

        xs = []
        for y in ys:
            mean = torch.tensor(means[y], dtype=torch.float32)
            covar = torch.tensor(covariances[y], dtype=torch.float32) 
            
            x = torch.distributions.MultivariateNormal(mean, covar)
            xs.append(x.sample()) ## of shape (num_points, dim_x)

        zipped = list(zip(xs, ys)) ## num_points each of form [list(x), int(y)]
        random.shuffle(zipped)
        contexts, targets = zipped[:nc], zipped[nc:]

        xc, yc = list(zip(*contexts))
        xt, yt = list(zip(*targets))

        return B.stack(*xc, axis=0), B.stack(*yc, axis=0), B.stack(*xt, axis=0), B.stack(*yt, axis=0) ## of form (n, c)


class gp_cutoff:
    '''
    A binary classifier with flexible distributions of points across each class' domain.
    Constructed by slicing a GP at a set distance from the extrema and only keeping points outside of it.

    Arguments:
    ----------------
    dim_x: (int) Dimension of inputs.
    xrange: (list[list, list]) Range for drawing of samples for domain. Of form [min_array, max_array] where min/max_array are dim_x dimensional lists.
    ...
    '''
    
    def __init__(self, dim_x, xrange, num_batches=16, nc_bounds=[10, 20], nt_bounds=[10, 20], kernel='eq', cutoff=None, device='cpu'):

        self.dim_x = dim_x
        self.xrange = xrange
        self.num_batches = num_batches
        self.kernel = kernel
        self.cutoff = cutoff
        self.nc_bounds = nc_bounds
        self.nt_bounds = nt_bounds
        self.device = device

        l = 2
        if kernel == 'eq': 
            f = lambda x1, x2: B.exp(-l * np.dot((x1 - x2), (x1-x2)) / 2)
        else: 
            print(f'Have not implemented {kernel} kernel, defaulting to EQ')
            f = lambda x1, x2: B.exp(-l * np.dot((x1 - x2), (x1-x2)) / 2)
        self.gram = lambda x: [[f(x1, x2) for x1 in x] for x2 in x]


    def _construct_gp_sample(self, xs, gram):
        gp_sample = lambda x: np.random.multivariate_normal(np.zeros(np.array(x).shape[0]), np.array(gram(x))) ## assumes mean 0 for gp
        return xs, gp_sample(xs)

    
    def _cutoff(self, xs, gp_sample, cutoff):
        '''
        Assign classes to xs based on whether ys are above or below the median value. Done if cutoff is None.
        Or:
        Cut-off the top and bottom 'cutoff' percent of points to be the classes and ignore the rest. NOTE: Not yet implemented.
        '''

        if not cutoff:
            _med = median(gp_sample)
            _sliced = list(map(lambda x,y: [x, float(y < _med)], xs, gp_sample))
        else:
            pass ##TODO: not yet implemented

        xs, ys = list(zip(*_sliced))
        return xs, ys


    def batch(self, gram, cutoff, nc_bounds, nt_bounds, xrange, dim_x):

        nc = random.randint(*nc_bounds)
        nt = random.randint(*nt_bounds)

        xs = np.random.uniform(low=xrange[0], high=xrange[1], size=(int(nc+nt), dim_x))
        xs, gp_sample = self._construct_gp_sample(xs, gram)
        xs, ys = self._cutoff(xs, gp_sample, cutoff)

        _zipped = list(zip(xs, ys))
        random.shuffle(_zipped)
        contexts, targets = _zipped[:nc], _zipped[nc:]

        xc, yc = list(zip(*contexts))
        xt, yt = list(zip(*targets))
        return xc, yc, xt, yt


    def epoch(self):

        with B.on_device(self.device):

            def convert_data(points):
                _xc, _yc, _xt, _yt = points
                xc = B.transpose(torch.tensor(np.array(_xc), dtype=torch.float32)) ## B.transpose defaults to switching the last 2 axes: (n, c) => (c, n)
                xt = B.transpose(torch.tensor(np.array(_xt), dtype=torch.float32))
                yc = B.expand_dims(torch.tensor(np.array(_yc), dtype=torch.float32), axis=0)
                yt = B.expand_dims(torch.tensor(np.array(_yt), dtype=torch.float32), axis=0)
                return xc, yc, xt, yt

            epoch = []        
            for _ in range(self.num_batches):
                _points = self.batch(self.gram, self.cutoff, self.nc_bounds, self.nt_bounds, self.xrange, self.dim_x)
                xc, yc, xt, yt = convert_data(_points)

                batch = {
                    'xc': xc.to(self.device),
                    'yc': yc.to(self.device),
                    'xt': xt.to(self.device),
                    'yt': yt.to(self.device),
                }
                epoch.append(batch)

            return epoch   