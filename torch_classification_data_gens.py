import lab as B
import torch
import random
import numpy as np
from statistics import median
from tqdm import tqdm, trange
from scipy.stats import multivariate_normal as norm


class example_data_gen:

    def __init__(self, means, covariances, num_batches = 1, batch_size = 16, dim_x: int = None, nc_bounds: list = [10, 20], nt_bounds: list = [10, 20], priors: list = [0.5, 0.5], device='cpu'):
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
        self.batch_size = batch_size


    def epoch(self):

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
            _xc, _yc, _xt, _yt, reference = points
            xc = B.transpose(_xc) ## B.transpose defaults to switching the last 2 axes: (n, c) => (c, n)
            xt = B.transpose(_xt)
            yc = B.cast(torch.float32, B.expand_dims(_yc, axis=0))
            yt = B.cast(torch.float32, B.expand_dims(_yt, axis=0))
            return xc, yc, xt, yt, reference

        epoch = []        
        nc = B.squeeze(torch.randint(low=self.nc_bounds[0], high=self.nc_bounds[1]+1, size=(1, )))
        nt = B.squeeze(torch.randint(low=self.nt_bounds[0], high=self.nt_bounds[1]+1, size=(1,)))

        for _ in range(self.num_batches):

            xc, yc, xt, yt, reference = [], [], [], [], []
            for _ in range(self.batch_size):
                _points = self._sub_batch(_means, _covars, self.dim_x, nc, nt, self.priors)
                sub_xc, sub_yc, sub_xt, sub_yt, sub_reference = convert_data(_points)

                xc.append(sub_xc)
                yc.append(sub_yc)
                xt.append(sub_xt)
                yt.append(sub_yt)
                reference.append(sub_reference)

            batch = {
                'xc': torch.stack(xc).to(self.device),
                'yc': torch.stack(yc).to(self.device),
                'xt': torch.stack(xt).to(self.device),
                'yt': torch.stack(yt).to(self.device),
                'reference': reference,
            }    
            epoch.append(batch)

        return epoch   


    def _sub_batch(self, means, covariances, dim_x, nc, nt, priors):
        
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

        xs = np.array(xs)
        temps = np.meshgrid(*[np.linspace(i, j, 30) for i, j in zip(np.min(xs, axis=0), np.max(xs, axis=0))])
        xref = np.dstack(temps)
        dist1 = norm(means[0], covariances[0])
        dist2 = norm(means[1], covariances[1])
        yref = (priors[0] * dist1.pdf(xref)) / (priors[0] * dist1.pdf(xref) + priors[1] * dist2.pdf(xref))
        reference = list(zip(xref, yref))

        return B.stack(*xc, axis=0), B.stack(*yc, axis=0), B.stack(*xt, axis=0), B.stack(*yt, axis=0), reference ## of form (n, c)


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
    
    def __init__(self, dim_x, xrange, batch_size=16, num_batches=1, nc_bounds=[10, 20], nt_bounds=[10, 20], kernel='eq', cutoff='median', device='cpu', reference=False):

        self.dim_x = dim_x
        self.xrange = xrange
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.kernel = kernel
        self.cutoff = cutoff
        self.nc_bounds = nc_bounds
        self.nt_bounds = nt_bounds
        self.device = device
        self.reference = reference

        l = (B.max(np.array(xrange[1])) - B.min(np.array(xrange[0]))) / 2 # this should be reflective of xrange!
        if kernel == 'eq': 
            f = lambda x1, x2: B.exp(- np.dot((x1 - x2), (x1-x2)) / (2*l))
        else: 
            print(f'Have not implemented {kernel} kernel, defaulting to EQ')
            f = lambda x1, x2: B.exp(- np.dot((x1 - x2), (x1-x2)) / (2*l))
        self.gram = lambda x: [[f(x1, x2) for x1 in x] for x2 in x]


    def _construct_gp_sample(self, xs, ref_xs, gram):

        gp_sample = lambda x: np.random.multivariate_normal(np.zeros(np.array(x).shape[0]), np.array(gram(x))) ## assumes mean 0 for gp

        if self.reference:
            concatenated = np.concatenate((xs, ref_xs), axis=0) # xs and ref_xs have shape (n, dim_x) here
            out = gp_sample(concatenated)
            ys, ref_ys = out[:len(xs)], out[len(xs):]
            return xs, ys, ref_ys
        
        else:
            return xs, gp_sample(xs), None

    
    def _cutoff(self, xs, gp_sample, cutoff):

        if cutoff == 'median':
            _med = median(gp_sample)
            _sliced = list(map(lambda x,y: [x, float(y < _med)], xs, gp_sample))
        elif cutoff == 'zero':
            _sliced = list(map(lambda x,y: [x, float(y < 0)], xs, gp_sample))

        xs, ys = list(zip(*_sliced))
        return xs, ys


    def _sub_batch(self, gram, cutoff, nc, nt, xrange, dim_x):

        xs = np.random.uniform(low=xrange[0], high=xrange[1], size=(int(nc+nt), dim_x))
        ref_xs = np.array(np.meshgrid(*[np.linspace(i, j, 30) for i, j in zip(xrange[0], xrange[1])])).reshape(dim_x, -1).T
        xs, gp_sample, ref_sample = self._construct_gp_sample(xs, ref_xs, gram)
        xs, ys = self._cutoff(xs, gp_sample, cutoff)

        if self.reference:
            ref_xs, ref_ys = self._cutoff(ref_xs, ref_sample, cutoff)
            ref = list(zip(ref_xs, ref_ys)) 
        else:
            ref = None

        zipped = list(zip(xs, ys))
        random.shuffle(zipped)
        contexts, targets = zipped[:nc], zipped[nc:]

        xc, yc = list(zip(*contexts))
        xt, yt = list(zip(*targets))
        return xc, yc, xt, yt, ref


    def epoch(self):

        epoch = []
        for _ in range(self.num_batches):

            def _convert_data(points):
                _xc, _yc, _xt, _yt, ref = points
                xc = B.transpose(torch.tensor(np.array(_xc), dtype=torch.float32)) ## B.transpose defaults to switching the last 2 axes: (n, c) => (c, n)
                xt = B.transpose(torch.tensor(np.array(_xt), dtype=torch.float32))
                yc = B.expand_dims(torch.tensor(np.array(_yc), dtype=torch.float32), axis=0)
                yt = B.expand_dims(torch.tensor(np.array(_yt), dtype=torch.float32), axis=0)
                return xc, yc, xt, yt, ref

            nc = random.randint(*self.nc_bounds)
            nt = random.randint(*self.nt_bounds)

            xc, yc, xt, yt, reference = [], [], [], [], []       
            for _ in trange(self.batch_size):

                _points = self._sub_batch(self.gram, self.cutoff, nc, nt, self.xrange, self.dim_x)
                sub_xc, sub_yc, sub_xt, sub_yt, sub_reference = _convert_data(_points)

                xc.append(sub_xc)
                yc.append(sub_yc)
                xt.append(sub_xt)
                yt.append(sub_yt)
                reference.append(sub_reference)

            batch = {
                'xc': torch.stack(xc).to(self.device),
                'xt': torch.stack(xt).to(self.device),
                'yc': torch.stack(yc).to(self.device),
                'yt': torch.stack(yt).to(self.device),
                'reference': reference,
            }
            epoch.append(batch)

        return epoch   