import torch
import numpy as np
from numpy import random
from tqdm import trange
import lab as B


class gp_example:
    

    def __init__(
        self,
        dim_x=2,
        num_batches=1,
        batch_size=16,
        device='cpu',
        nc_bounds=[50, 100], 
        nt_bounds=[50, 100],
        xrange=[[0, 0], [60, 60]],
        kernel='eq',
        reference=False,
    ):

        self.dim_x = dim_x
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device
        self.nc_bounds = nc_bounds
        self.nt_bounds = nt_bounds
        self.reference = reference
        self.xrange = xrange

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
            return xs, B.exp(ys), B.exp(ref_ys)
        
        else:
            return xs, B.exp(gp_sample(xs)), None


    def _sub_batch(self, gram, nc, nt, xrange, dim_x):

        xs = np.random.uniform(low=xrange[0], high=xrange[1], size=(int(nc+nt), dim_x))
        ref_xs = np.array(np.meshgrid(*[np.linspace(i, j, 30) for i, j in zip(xrange[0], xrange[1])])).reshape(dim_x, -1).T
        xs, ys, ref_ys = self._construct_gp_sample(xs, ref_xs, gram)

        if self.reference:
            ref = list(zip(ref_xs, ref_ys)) 
        else:
            ref = None

        zipped = list(zip(xs, ys))
        random.shuffle(zipped)
        contexts, targets = zipped[:nc], zipped[nc:]

        xc, yc = list(zip(*contexts))
        xt, yt = list(zip(*targets))

        sub_batch = {
            'xc': torch.tensor(xc, dtype=torch.float32).T.to(self.device),
            'yc': torch.tensor(yc, dtype=torch.float32).reshape((1, -1)).to(self.device),
            'xt': torch.tensor(xt, dtype=torch.float32).T.to(self.device),
            'yt': torch.tensor(yt, dtype=torch.float32).reshape((1, -1)).to(self.device),
            'reference': ref,
        }

        return sub_batch


    def epoch(self):

        epoch = []
        for _ in range(self.num_batches):

            nc = random.randint(*self.nc_bounds)
            nt = random.randint(*self.nt_bounds)
            self.nc, self.nt = nc, nt

            xc, yc, xt, yt, reference = [], [], [], [], []
            for _ in trange(self.batch_size):
                sub_batch = self._sub_batch(self.gram, nc, nt, self.xrange, self.dim_x)
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


class synthetic_gamma_reader:


    def __init__(self, num_batches = 1, batch_size = 16, starting_ind = 0, device= 'cpu'):
        self.path = 'real_data/'
        self.i = starting_ind
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device

    
    def epoch(self):

        epoch = []
        for _ in range(self.num_batches):

            xc, yc, xt, yt, reference = [], [], [], [], []
            for _ in range(self.batch_size):

                try:
                    torch.load(self.path + f'real-{self.i}.tensor')
                except:
                    print(f'Ran out of tasks to load. {self.batch_size*self.num_batches} is greater than number of tasks. Will repeat tasks.')
                    self.i = 0

                read_task = torch.load(self.path + f'real-{self.i}.tensor')
                yc_gamma = read_task['yc_gamma']
                yt_gamma = read_task['yt_gamma']
                xc_gamma = read_task['xc']
                xt_gamma = read_task['xt']

                filt_xc_gamma, filt_yc_gamma = [], []
                for i, j in zip(xc_gamma.reshape(-1, 2), yc_gamma.reshape(-1, 1)):
                    if j > 0:
                        filt_xc_gamma.append(i)
                        filt_yc_gamma.append(j)

                filt_xt_gamma, filt_yt_gamma = [], []
                for i, j in zip(xt_gamma.reshape(-1, 2), yt_gamma.reshape(-1, 1)):
                    if j > 0:
                        filt_xt_gamma.append(i)
                        filt_yt_gamma.append(j)

                try:
                    xc_gamma = torch.stack(filt_xc_gamma).reshape(2, -1)
                    yc_gamma = torch.stack(filt_yc_gamma).reshape(1, -1)
                    xt_gamma = torch.stack(filt_xt_gamma).reshape(2, -1)
                    yt_gamma = torch.stack(filt_yt_gamma).reshape(1, -1)
                except RuntimeError:
                    xc_gamma = xc_gamma.reshape(2, -1)
                    yc_gamma = torch.zeros(xc_gamma.shape).reshape(1, -1)
                    xt_gamma = xt_gamma.reshape(2, -1)
                    yt_gamma = torch.zeros(xt_gamma.shape).reshape(1, -1)

                xc.append(xc_gamma)
                xt.append(xt_gamma)
                yc.append(yc_gamma)
                yt.append(yt_gamma)
                reference.append(read_task['reference'])
                
                self.i += 1

            min_len_c = min([i.shape[1] for i in xc])
            min_len_t = min([i.shape[1] for i in xt])
            xc = [i[:, :min_len_c] for i in xc]
            xt = [i[:, :min_len_t] for i in xt]
            yc = [i[:, :min_len_c] for i in yc]
            yt = [i[:, :min_len_t] for i in yt]

            batch = {
                'xc': torch.stack(xc).to(self.device),
                'xt': torch.stack(xt).to(self.device),
                'yc': torch.stack(yc).to(self.device),
                'yt': torch.stack(yt).to(self.device),
                'reference': reference,
            }

            epoch.append(batch)

        return epoch