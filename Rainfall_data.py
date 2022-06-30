import lab as B
import torch
import numpy as np
from typing import Tuple
import random
from statistics import median
from datetime import datetime, timedelta
import pandas as pd
import shutup
shutup.please()


class rainfall_generator:

    def __init__(
        self, 
        batch_size, 
        nc_bounds: Tuple[int, int] = [50, 70],
        nt_bounds: Tuple[int, int] = [100, 140],
        include_binary: bool = False,
        device='cpu'
        ):
        self.batch_size = batch_size
        self.device = device
        self.include_binary = include_binary

        self.folder = 'C:/Users/bened/Documents/University/Cambridge/Thesis/Data/NordicDailyPrecip/'
        self.nc = random.randint(*nc_bounds)
        self.nt = random.randint(*nt_bounds)
        self.days = np.random.choice(np.arange(365), size=self.batch_size, replace=False)


    def _to_int(self, date):
            return 10000*date.year + 100*date.month + date.day


    def batch(self, day_ind):

        _start = datetime(2018, 1, 1)
        _d = self.days[day_ind] 
        _day = _start + timedelta(days=int(_d))
        date = self._to_int(_day)
        df = pd.read_csv(self.folder + f'Agg(max)_Reduced_{date}.csv', usecols=['X', 'Y', 'RR', 'Binary'])

        reference = df['RR'].to_numpy().reshape(np.sqrt(df['RR'].shape[0]).astype(int), np.sqrt(df['RR'].shape[0]).astype(int)) # assumes square datagrid
                
        _select_inds = np.random.choice(len(df['RR']), size=int(self.nc + self.nt), replace=False)
        _x1s = df['X'].iloc[_select_inds].to_numpy()
        _x2s = df['Y'].iloc[_select_inds].to_numpy()
        _precip = df['RR'].iloc[_select_inds].to_numpy()
        _binary = df['Binary'].iloc[_select_inds].to_numpy()

        xs = list(zip(_x1s, _x2s))
        xc, xt = np.array(xs[:self.nc]).T, np.array(xs[self.nc:]).T # to shape [2, nc(nt)]
        if not self.include_binary:
            yc, yt = _precip[:self.nc].reshape(1, -1), _precip[self.nc:].reshape(1, -1) # to shape [1, nc(nt)]
        else:
            yc, yt = [_precip[:self.nc], _binary[:self.nc]], [_precip[self.nc:], _binary[self.nc:]] # to shape [2, nc(nt)]

        batch = {
            'xc': torch.tensor(xc, dtype=torch.float32).to(self.device),
            'yc': torch.tensor(yc, dtype=torch.float32).to(self.device),
            'xt': torch.tensor(xt, dtype=torch.float32).to(self.device),
            'yt': torch.tensor(yt, dtype=torch.float32).to(self.device),
            'reference': reference,
            'date': date,
        }
        return batch


    def epoch(self):
        '''
        Return an epoch's worth of data in the form (b, c, n) for all of xc, yc, xt, yt (in dict form). 
        Note that epoch = [batch1, ...] and batchi = {'xc': ..., 'yc': ..., ...} by convention.
        '''

        epoch = []
        for i in range(self.batch_size):
            epoch.append(self.batch(i))

        return epoch


class Bernoulli_Gamma_synthetic:

    def __init__(
        self, 
        batch_size: int, 
        xrange: Tuple[float, float], 
        nc_bounds: Tuple[int, int] = [50, 70], 
        nt_bounds: Tuple[int, int] = [100, 140], 
        device: str = 'cpu',
        kernel: str = 'eq',
        l: float = 1.,
        gp_mean: float = 0.,
        num_ref_points: int = None,
        include_binary: bool = False,
        ):

        self.batch_size = batch_size
        self.xrange = xrange
        self.device = device
        self.nc_bounds = nc_bounds
        self.nt_bounds = nt_bounds
        self.gp_mean = gp_mean
        self.num_ref_points = num_ref_points
        self.include_binary = include_binary

        if kernel == 'eq': 
            f = lambda x1, x2: B.exp(-1 * np.dot((x1 - x2), (x1-x2)) / 2*(l**2))
        elif kernel == 'laplace':
            f = lambda x1, x2: B.exp(-1 * np.dot((x1 - x2), (x1-x2))**0.5 / l)
        else: 
            print(f'Have not implemented {kernel} kernel, defaulting to EQ')
            f = lambda x1, x2: B.exp(-l * np.dot((x1 - x2), (x1-x2)) / 2)
        self.gram = lambda x: [[f(x1, x2) for x1 in x] for x2 in x]


    def _construct_gp_sample(self, xs, gram):
        gp_sample = lambda x: np.random.multivariate_normal(self.gp_mean * np.ones(np.array(x).shape[0]), np.array(gram(x))) ## assumes mean 0 for gp
        return xs, gp_sample(xs)

    
    def _cutoff(self, xs, gp_sample):
        '''
        If gp_sample < 0: y = 0. If gp_sample > 0: y = gp_sample
        '''
        _med = median(gp_sample)
        _sliced = list(map(lambda x,y: [x, 0. if y < _med else y], xs, gp_sample))
        xs, ys = list(zip(*_sliced))
        return xs, ys


    def batch(self):

        nc = random.randint(*self.nc_bounds)
        nt = random.randint(*self.nt_bounds)

        xs = np.random.uniform(low=self.xrange[0], high=self.xrange[1], size=(int(nc+nt), 2))
        xs, gp_sample = self._construct_gp_sample(xs, self.gram)
        xs, ys = self._cutoff(xs, gp_sample)

        if self.include_binary:
            _binary = [float(i != 0.) for i in ys]
            ys = list(zip(ys, _binary))

        if self.num_ref_points:
            xref = []
            for i in np.linspace(*self.xrange, self.num_ref_points):
                for j in np.linspace(*self.xrange, self.num_ref_points):
                    xref.append([i, j])
            _, reference = self._construct_gp_sample(np.array(xref), self.gram)
            reference = reference.reshape(self.num_ref_points, self.num_ref_points)
        else:
            reference = None

        _zipped = list(zip(xs, ys))
        random.shuffle(_zipped)
        contexts, targets = _zipped[:nc], _zipped[nc:]

        xc, yc = list(zip(*contexts))
        xt, yt = list(zip(*targets))

        batch = {
            # NOTE: use of .T on tensor depracated and will cause errors eventually.
            'xc': torch.tensor(xc, dtype=torch.float32).T.to(self.device),
            'yc': torch.tensor(yc, dtype=torch.float32).T.to(self.device),
            'xt': torch.tensor(xt, dtype=torch.float32).T.to(self.device),
            'yt': torch.tensor(yt, dtype=torch.float32).T.to(self.device),
            'reference': reference,
        }
        return batch


    def epoch(self):
        '''
        Return an epoch's worth of data in the form (b, c, n) for all of xc, yc, xt, yt (in dict form). 
        Note that epoch = [batch1, ...] and batchi = {'xc': ..., 'yc': ..., ...} by convention.
        '''

        epoch = []
        for _ in range(self.batch_size):
            epoch.append(self.batch())

        return epoch


if __name__ == '__main__':

    synthetic = Bernoulli_Gamma_synthetic(
        xrange=[0, 20],
        batch_size=1,
        nc_bounds=[50, 70],
        nt_bounds=[100, 140],
        device='cpu',
        kernel='eq',
        l=0.2,
        gp_mean=1,
        num_ref_points=30, # cost of creaeting each batch scales with O(num_ref_points^2)
    )

    rainfall = rainfall_generator(
        batch_size=1,
        nc_bounds=[50, 70],
        nt_bounds=[100, 140],
        device='cpu',
    )

    import matplotlib.pyplot as plt

    batch = rainfall.epoch()[0]
    print(batch['date'])

    xc = batch['xc']
    yc = batch['yc']
    print(xc.shape, yc.shape)
    print(batch['reference'].shape)

    plt.imshow(batch['reference'], vmin=0, cmap='Pastel2') # 'turbo' cmap
    plt.colorbar()
    plt.show()