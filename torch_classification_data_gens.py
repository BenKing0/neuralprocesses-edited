import lab as B
import torch
import random


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