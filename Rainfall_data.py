import lab as B
import torch
import numpy as np
from typing import Tuple
import random
from statistics import median
from datetime import datetime, timedelta
import pandas as pd
from sys import getsizeof
import shutup
from tqdm import tqdm, trange
from scipy.stats import multivariate_normal as norm
shutup.please()
import time
t = int(1000 * time.time()) % 2**32
np.random.seed(t)
torch.random.manual_seed(t)


class rainfall_generator:

    def __init__(
        self, 
        batch_size, 
        num_batches: int = 1,
        nc_bounds: Tuple[int, int] = [50, 70],
        nt_bounds: Tuple[int, int] = [100, 140],
        include_binary: bool = True,
        device='cpu'
        ):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.include_binary = include_binary

        self.folder = 'data/'
        self.nc = random.randint(*nc_bounds)
        self.nt = random.randint(*nt_bounds)
        self.days = np.random.choice(np.arange(365), size=self.batch_size, replace=False)


    def _to_int(self, date):
            return 10000*date.year + 100*date.month + date.day


    def _sub_batch(self, day_ind):

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

        sub_batch = {
            'xc': torch.tensor(xc, dtype=torch.float32).to(self.device),
            'yc': torch.tensor(yc, dtype=torch.float32).to(self.device),
            'xt': torch.tensor(xt, dtype=torch.float32).to(self.device),
            'yt': torch.tensor(yt, dtype=torch.float32).to(self.device),
            'reference': reference,
            'date': date,
        }
        return sub_batch


    def epoch(self):
        '''
        Return an epoch's worth of data in the form (b', b, c, n) for all of xc, yc, xt, yt (in dict form), where b' is num_batches, and b is batch_size
        Note that epoch = [batch1, ...] and batchi = {'xc': ..., 'yc': ..., ...} by convention.
        '''

        epoch = []
        for _ in range(self.num_batches):

            xc, yc, xt, yt, reference, dates = [], [], [], [], [], []
            for j in range(self.batch_size):
                sub_batch = self._sub_batch(j)
                xc.append(sub_batch['xc'])
                xt.append(sub_batch['xt'])
                yc.append(sub_batch['yc'])
                yt.append(sub_batch['yt'])
                reference.append(sub_batch['reference'])
                dates.append(sub_batch['date'])

            batch = {
                # each of shape (b, c, n) where b=batch_size and c=2 for xs and ys:
                'xc': torch.stack(xc).to(self.device),
                'yc': torch.stack(yc).to(self.device),
                'xt': torch.stack(xt).to(self.device),
                'yt': torch.stack(yt).to(self.device),
                'reference': reference,
                'date': dates,
            }

            epoch.append(batch)

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
        include_binary: bool = True,
        num_batches: int = 1,
        ):

        self.batch_size = batch_size
        self.num_batches = num_batches
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


    def _construct_gp_sample(self, xs, ref_xs, gram):
        gp_sample = lambda x: np.random.multivariate_normal(self.gp_mean * np.ones(np.array(x).shape[0]), np.array(gram(x))) ## assumes mean 0 for gp
        
        if self.num_ref_points:
            concatenated = np.concatenate((xs, ref_xs), axis=0) # xs and ref_xs have shape (n, dim_x) here
            out = gp_sample(concatenated)
            ys, ref_ys = out[:len(xs)], out[len(xs):]
            return xs, ys, ref_ys
        
        else:
            return xs, gp_sample(xs), None

    
    def _cutoff(self, xs, gp_sample):
        '''
        If gp_sample < 0: y = 0. If gp_sample > 0: y = gp_sample
        '''
        _sliced = list(map(lambda x,y: [x, 0. if y < 0. else y], xs, gp_sample))
        xs, ys = list(zip(*_sliced))
        return xs, ys

    
    def _sub_batch(self):

        xs = np.random.uniform(low=self.xrange[0], high=self.xrange[1], size=(int(self.nc+self.nt), 2))
        ref_xs = np.array(np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], 30), np.linspace(self.xrange[0], self.xrange[1], 30))).reshape(2, -1).T
        xs, gp_sample, ref_sample = self._construct_gp_sample(xs, ref_xs, self.gram)
        xs, ys = self._cutoff(xs, gp_sample)

        if self.num_ref_points:
            ref_xs, ref_ys = self._cutoff(ref_xs, ref_sample)
            reference = np.array(ref_ys).reshape(self.num_ref_points, self.num_ref_points)
        else:
            reference = None

        if self.include_binary:
            _binary = [float(i != 0.) for i in ys]
            ys = list(zip(ys, _binary))

        _zipped = list(zip(xs, ys))
        random.shuffle(_zipped)
        contexts, targets = _zipped[:self.nc], _zipped[self.nc:]

        xc, yc = list(zip(*contexts))
        xt, yt = list(zip(*targets))

        sub_batch = {
            # NOTE: use of .T on tensor depracated and will cause errors eventually.
            'xc': torch.tensor(np.array(xc), dtype=torch.float32).T.to(self.device),
            'yc': torch.tensor(np.array(yc), dtype=torch.float32).T.to(self.device),
            'xt': torch.tensor(np.array(xt), dtype=torch.float32).T.to(self.device),
            'yt': torch.tensor(np.array(yt), dtype=torch.float32).T.to(self.device),
            'reference': reference,
        }
        return sub_batch


    def epoch(self):
        '''
        Return an epoch's worth of data in the form (b', b, c, n) for all of xc, yc, xt, yt (in dict form), where b' is num_batches, and b is batch_size
        Note that epoch = [batch1, ...] and batchi = {'xc': ..., 'yc': ..., ...} by convention.
        '''

        epoch = []
        for _ in range(self.num_batches):

            nc = random.randint(*self.nc_bounds)
            nt = random.randint(*self.nt_bounds)
            self.nc, self.nt = nc, nt

            xc, yc, xt, yt, reference = [], [], [], [], []
            for _ in trange(self.batch_size):
                sub_batch = self._sub_batch()
                xc.append(sub_batch['xc'])
                xt.append(sub_batch['xt'])
                yc.append(sub_batch['yc'])
                yt.append(sub_batch['yt'])
                reference.append(sub_batch['reference'])

            batch = {
                # each of shape (b, c, n) where b=batch_size and c=2 for xs and ys:
                'xc': torch.stack(xc).to(self.device),
                'yc': torch.stack(yc).to(self.device),
                'xt': torch.stack(xt).to(self.device),
                'yt': torch.stack(yt).to(self.device),
                'reference': reference,
            }

            epoch.append(batch)

        return epoch


if __name__ == '__main__':

    import pandas as pd
    import matplotlib.pyplot as plt

    num_batches = 1
    batch_size = 365

    synthetic = Bernoulli_Gamma_synthetic(
        xrange=[0, 60],
        batch_size=batch_size,
        nc_bounds=[200, 300],
        nt_bounds=[100, 150],
        device='cpu',
        kernel='eq',
        l=0.125,
        gp_mean=0,
        num_ref_points=30, # cost of creating each batch scales with O(num_ref_points^2)
    )

    real = rainfall_generator(
        batch_size=batch_size,
        nc_bounds=[200, 300],
        nt_bounds=[100, 150],
        device='cpu',
    )

    i = 0
    for batch in range(num_batches):

        batch = real.epoch()[0]

        for task in range(batch_size):

            # each (c, n)
            xc = batch['xc'][task]
            yc = batch['yc'][task]
            xt = batch['xt'][task]
            yt = batch['yt'][task]
            reference = batch['reference'][task].T

            yc_bernoulli = yc[1:2] # (1, n)
            yt_bernoulli = yt[1:2]

            yc_gamma = yc[0:1]
            yt_gamma = yc[0:1]

            save_batch = {
                'xc': xc,
                'xt': xt,
                'yc_bernoulli': yc_bernoulli,
                'yt_bernoulli': yt_bernoulli,
                'yc_gamma': yc_gamma,
                'yt_gamma': yt_gamma,
                'reference': reference,
            }

            torch.save(save_batch, f'real_data/real-{i}.tensor')

            i += 1
