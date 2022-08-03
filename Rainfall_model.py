from ctypes import sizeof
import re
# from statistics import covariance
from neuralprocesses import aggregate
from neuralprocesses.aggregate import Aggregate
from neuralprocesses.disc import Discretisation
from neuralprocesses.torch import Parallel
from neuralprocesses.util import batch
import shutup
from sklearn.metrics import mean_poisson_deviance
import neuralprocesses.torch as nps
import neuralprocesses as base
import torch
import lab.torch as B
import wbml.out as out
import experiment as exp
import numpy as np
from plum import convert
from functools import partial
from wbml.experiment import WorkingDirectory
from Rainfall_data import rainfall_generator, Bernoulli_Gamma_synthetic, bernoulli_only_GP, bernoulli_only_MoG
from Rainfall_plotting import rainfall_plotter
from typing import List
from neuralprocesses.model.elbo import _merge_context_target
import time
shutup.please()

t = int(time.time() * 1000) % 2**32
torch.random.manual_seed(t)
np.random.seed(t)

# TODO: The issue is that the parameters are not being updated correctly!


# `Aggregate` outputs are assumed to always come with `AggregateInput` inputs.
# In this case, that's not true, so add support for this.

@_merge_context_target.dispatch
def _merge_context_target(contexts: List, xt: B.Numeric, yt: base.Aggregate):
    return _merge_context_target(
        contexts,
        nps.AggregateInput(*((xt, i) for i in range(len(yt)))),
        yt,
    )


class BernoulliDistribution:

    def __init__(self, probs):
        self.probs = probs # (*b, 1, n)
        print(f'\nProb range: {float(torch.min(self.probs.flatten())):.2f} to {float(torch.max(self.probs.flatten())):.2f}')

    def logpdf(self, y): 
        # NOTE: y of shape (b, 1, n) as constant across samples so no sample dimension. Broadcasting stretches copies of y to same shape automatically. 
        # y = torch.stack(*[y for _ in self.probs.shape[0]])

        print(f'Proportion 1s: {torch.sum(y) / torch.sum(torch.ones(y.shape)):.2f}')

        return B.sum(
            B.log(self.probs) * y + B.log(1 - self.probs) * (1 - y),
            axis=(-2, -1),
        )
                

class GammaDistribution:

    def __init__(self, params, device):
        self.kappa = torch.ones(params[..., 0:1, :].shape).to(device) + params[..., 0:1, :] # shape (*b, c=2, n) -> (*b, 1, n)
        self.chi = B.log(1e-3 + B.exp(params[..., 1:2, :])) # shape (*b, c=2, n) -> (*b, 1, n)

    def logpdf(self, y_rain, y_amount):
        # NOTE: each of shape (b, 1, n) whereas kappa, chi possibly of shape (s, b, 1, n), but broadcasting is automatic.

        print(f'Kappa range: {torch.min(self.kappa.flatten()):.2f} to {torch.max(self.kappa.flatten()):.2f}')
        print(f'Chi range: {torch.min(self.chi.flatten()):.2f} to {torch.max(self.chi.flatten()):.2f}\n')

        # TODO: what makes the gamma cost nan? Not the factors above!
        return B.sum(
            ((self.kappa - 1) * torch.nan_to_num(torch.log(y_amount), 0.) - self.chi * y_amount + self.kappa * torch.log(self.chi) - torch.lgamma(self.kappa)) * y_rain,
            axis=(-2, -1),
        )


class BernoulliGammaDist:

    def __init__(self, z_bernoulli, z_gamma, device):

        # These are the model predictions for bernoulli statistic and gamma statistics. (*b, c, n) where c=1 for bernouuli and c=2 for gamma
        self.bernoulli_prob = B.sigmoid(z_bernoulli)
        self.z_gamma = B.softplus(z_gamma)
        self.device = device

    def logpdf(self, ys):

        # Next 3 lines unaggregate the Aggregate ys into a stack of torch tensors via a list (of shape (*b, 2, n))
        ys = torch.stack([el for el in ys])
        ys = torch.tensor(ys, dtype=torch.float32)
        y_rain = ys[0] # (b, 1, n)
        y_amount = ys[1] # (b, 1, n)

        assert len(torch.unique(y_rain.flatten())) == 2

        bernoulli_dist = BernoulliDistribution(self.bernoulli_prob)
        bernoulli_cost = bernoulli_dist.logpdf(y=y_rain)
        gamma_dist = GammaDistribution(self.z_gamma, device=self.device) 
        gamma_cost = 0*gamma_dist.logpdf(y_amount=y_amount, y_rain=y_rain)

        print(f'Mean Bernoulli cost: {torch.mean(bernoulli_cost.flatten()):.2f}, Mean Gamma cost: {torch.mean(gamma_cost.flatten()):.2f}\n')

        return torch.tensor(bernoulli_cost + gamma_cost, dtype=torch.float32) # (*b,) = (num_samples, num_batches)


class combined:

    @classmethod
    def combined_model(
        cls,
        discretisation,
        arch,
        dim_lv,
        num_layers,
        encoder_num_channels,
        decoder_num_channels,
        device,
        ):

        if arch == 'unet':
            decoder_channels = (encoder_num_channels,) * num_layers
            encoder_channels = (decoder_num_channels,) * num_layers

            net_latent_variable = nps.UNet(
                dim=2,
                in_channels=4,
                out_channels=dim_lv * (2 + 64),
                channels=encoder_channels,
                separable=False,
            )
            
            net = nps.UNet(
                dim=2,
                in_channels=dim_lv,
                out_channels=3,
                channels=decoder_channels,
                separable=False,
            )

        elif arch == 'convnet':

            decoder_channels = encoder_num_channels
            encoder_channels = decoder_num_channels

            net_latent_variable = nps.ConvNet(
                dim=2,
                in_channels=4,
                out_channels=dim_lv * (2 + 64),
                channels=encoder_channels,
                separable=False,
                num_layers=num_layers,
                points_per_unit=discretisation,
                receptive_field=4,
            )
            
            net = nps.ConvNet(
                dim=2,
                in_channels=dim_lv,
                out_channels=3,
                channels=decoder_channels,
                separable=False,
                num_layers=num_layers,
                points_per_unit=discretisation,
                receptive_field=4,
            )

        model = nps.Model(
            nps.FunctionalCoder(
                nps.Discretisation(
                    points_per_unit=discretisation,
                    multiple=2**net.num_halving_layers,
                    margin=0.1,
                ),
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.Parallel(
                        nps.SetConv(scale=2 / discretisation),
                        nps.SetConv(scale=2 / discretisation),
                    ),
                    nps.DivideByFirstChannel(),
                    nps.Concatenate(),
                    net_latent_variable,
                    nps.LowRankGaussianLikelihood(64),
                ),
            ),
            nps.Chain(
                net,
                nps.SetConv(scale=1 / discretisation),
                nps.Splitter(1, 2),
                lambda xs: BernoulliGammaDist(*xs, device=device), # (*b, c, n) for each of bernoulli (c=1) and gamma (c=2)
            ),
        )

        return model


    @classmethod
    def construct_model(
        cls,
        lv_likelihood: str,
        discretisation: int,
        dim_lv: int,
    ):

        bernoulli_model = cls._bernoulli_model(lv_likelihood=lv_likelihood, dim_lv=dim_lv, discretisation=discretisation)
        gamma_model = cls._gamma_model(lv_likelihood=lv_likelihood, dim_lv=dim_lv, discretisation=discretisation)

        pipeline = Parallel(bernoulli_model, gamma_model) # when __call__(x) done on a Parrallel, it returns Parrallel(*[el(x) for el in elements]) where Parallel(*elements) is the instantiation

        return pipeline # model to pass to elbo. NOTE: Adapted elbo.py to handle parallel.encoder and parallel.decoder


def train(state, model, opt, objective, gen, *, epoch):
    """Train for an epoch."""
    vals = []
    batches = gen.epoch()
    for i, batch in enumerate(batches):
        if i == 0: print('data gen device:', B.device(batch['xc']))
        # must all be of shape (*b, c, n) where c=2 for both yc(t) and xc(t), and c=1 for yc(t)_bernoulli(precip)
        xc = batch['xc']
        yc_bernoulli, yc_precip = batch['yc'][..., 1:2, :], batch['yc'][..., 0:1, :]
        yt_bernoulli, yt_precip = batch['yt'][..., 1:2, :], batch['yt'][..., 0:1, :]
        assert len(torch.unique(yc_bernoulli.flatten())) == 2
        obj = objective(
                model,
                [(xc, yc_bernoulli), (xc, yc_precip)],
                batch["xt"],
                nps.Aggregate(yt_bernoulli, yt_precip),
            )
        vals.append(B.to_numpy(obj))
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()
    
    out.kv("Loglik (T)", exp.with_err(B.concat(*vals)))
    return state, B.mean(B.concat(*vals))


def eval(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():
        vals = []
        batches = gen.epoch()
        for batch in batches:
            xc = batch['xc']
            yc_bernoulli, yc_precip = batch['yc'][..., 1:2, :], batch['yc'][..., 0:1, :]
            yt_bernoulli, yt_precip = batch['yt'][..., 1:2, :], batch['yt'][..., 0:1, :]
            assert len(torch.unique(yc_bernoulli.flatten())) == 2
            obj = objective(
                    model,
                    [(xc, yc_bernoulli), (xc, yc_precip)],
                    batch["xt"],
                    nps.Aggregate(yt_bernoulli, yt_precip),
                )
            vals.append(B.to_numpy(obj))

        out.kv("Loglik (V)", exp.with_err(B.concat(*vals)))
        return state, B.mean(B.concat(*vals))


def main(config, _config):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    B.set_global_device(device)
    print('global device: ', B.ActiveDevice.active_name)
    print('model device: ', device)

    nc_bounds = config.nc_bounds
    nt_bounds = config.nt_bounds
    batch_size = config.num_batches

    if config.data == 'rainfall':               
        gen_train, gen_cv, gens_eval = [    
            rainfall_generator(batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, include_binary=True, device=device),
            rainfall_generator(batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, include_binary=True,  device=device),
            rainfall_generator(batch_size=1, nc_bounds=nc_bounds, nt_bounds=nt_bounds, include_binary=True,  device=device),
            ]
    elif config.data == 'bernoulli_only_GP':               
        gen_train, gen_cv, gens_eval = [    
            bernoulli_only_GP(batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, reference=True, device=device),
            bernoulli_only_GP(batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, reference=True,  device=device),
            bernoulli_only_GP(batch_size=1, nc_bounds=nc_bounds, nt_bounds=nt_bounds, reference=True,  device=device),
        ]
    elif config.data == 'bernoulli_only_MoG': 
        means = [[10, 10] , [50, 50]]
        covariances = [5*np.eye(2), 5*np.eye(2)]
        dim_x = 2              
        gen_train, gen_cv, gens_eval = [    
            bernoulli_only_MoG(means=means, covariances=covariances, dim_x=dim_x, batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device),
            bernoulli_only_MoG(means=means, covariances=covariances, dim_x=dim_x, batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device),
            bernoulli_only_MoG(means=means, covariances=covariances, dim_x=dim_x, batch_size=1, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device),
        ]
    else:
        gen_train, gen_cv, gens_eval = [    
            Bernoulli_Gamma_synthetic(xrange=[0, 60], batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device, kernel='eq', l=0.2, gp_mean=1, num_ref_points=30, include_binary=True),
            Bernoulli_Gamma_synthetic(xrange=[0, 60], batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device, kernel='eq', l=0.2, gp_mean=1, num_ref_points=30, include_binary=True),
            Bernoulli_Gamma_synthetic(xrange=[0, 60], batch_size=1, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device, kernel='eq', l=0.2, gp_mean=1, num_ref_points=30, include_binary=True),
        ]

    if config.type == "combined":
        model = combined.combined_model(discretisation=1,
        arch=config.arch, 
        encoder_num_channels=config.encoder_channels,
        decoder_num_channels=config.decoder_channels,
        num_layers=config.num_layers,
        dim_lv=config.dim_lv,
        device=device,
        )
               

    elif config.type == 'separate':
        model = separate.construct_model(
            dim_lv = config.dim_lv,
            lv_likelihood = config.lv_likelihood,
            discretisation = config.discretisation,
        )

    model = model.to(device)
    out.kv("Number of parameters", nps.num_params(model))

    state = B.create_random_state(torch.float32)

    out.report_time = True
    B.epsilon = 1e-8
    wd = WorkingDirectory(
        *config.root,
        *(config.subdir or ()),
        config.data,
        *((f"x{config.dim_x}_y{config.dim_y}",) if hasattr(config, "dim_x") else ()),
        config.model,
        *((config.arch,) if hasattr(config, "arch") else ()),
        config.objective,
        log=f"log{config.mode}.txt",
        diff=f"diff{config.mode}.txt",
    )

    objective = partial(
            nps.elbo,
            num_samples=config.num_samples,
        )
    objective_cv = partial(
            nps.elbo,
            num_samples=config.num_samples,
        )
    objective_eval = partial(
            nps.elbo,
            num_samples=5,
        )
        
    ## In an evaluation regime only:
    if config.evaluate:

        if config.evaluate_last:
            name = "model-last.torch"
        else:
            name = "model-best.torch"
        model.load_state_dict(torch.load(wd.file(name), map_location=device)["weights"])

        rainfall_plotter(
            state=state, 
            model=model,
            save_path=wd.file(f"evaluate-{config.data}.png"),
            generator=gens_eval, 
            xbounds=[0, 60],
            ybounds=[0, 60],
            reference=True,
            device=device,
            )

        with out.Section('ELBO'):
            state, _ = eval(state, model, objective_eval, gen_cv)

    ## In a training (and cv) regime:
    else:

        start = 0
        if config.resume_at_epoch:
            start = config.resume_at_epoch - 1
            model.load_state_dict(
                torch.load(wd.file("model-last.torch"), map_location=device)["weights"]
            )

        opt = torch.optim.Adam(model.parameters(), config.rate)
        best_eval_lik = -np.inf

        # Set regularisation high for the first epochs.
        original_epsilon = B.epsilon
        B.epsilon = 1e-2

        for i in range(start, config.epochs):
            with out.Section(f"Epoch {i + 1}"):
                # Set regularisation to normal after the first epoch.
                if i > 0:
                    B.epsilon = original_epsilon

                # Perform an epoch.
                state, _ = train(
                    state,
                    model,
                    opt,
                    objective,
                    gen_train,
                    epoch=i if config.fix_noise else None,
                )

                state, val = eval(state, model, objective_cv, gen_cv)

                torch.save(
                    {
                        "weights": model.state_dict(),
                        "objective": val,
                        "epoch": i + 1,
                    },
                    wd.file(f"model-last.torch"),
                )

                rainfall_plotter(
                    state=state, 
                    model=model,
                    save_path=wd.file(f"train-epoch-{i+1}-{config.data}.png"),
                    generator=gens_eval, 
                    xbounds=[0, 60],
                    ybounds=[0, 60],
                    reference=True,
                    device=device,
                    )

                if val > best_eval_lik:
                    out.out("New best model!")
                    best_eval_lik = val
                    torch.save(
                        {
                            "weights": model.state_dict(),
                            "objective": val,
                            "epoch": i + 1,
                        },
                        wd.file(f"model-best.torch"),
                    )


if __name__ == '__main__':

    class dict2dot(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    _config = {
        "type": "combined", # NOTE: 'seperate' is not yet operational
        "arch": 'unet',
        "objective": 'elbo',
        "model": 'rainfall',
        "dim_x": 2, # NOTE: Hard-coded, included for filename (Has to be the case for rainfall case)
        "dim_y": 1, # NOTE: Hard-coded, included for filename (Has to be the case for rainfall case)
        "dim_lv": 16, # TODO: is high LV dim detramental?
        "data": 'bernoulli_only_MoG',
        "lv_likelihood": 'lowrank',
        "root": ["_experiments"],
        "epochs": 100,
        "train_test": None,
        "evaluate": False,
        "rate": 1e-3,
        "evaluate_last": False,
        "evaluate_num_samples": 20,
        "num_samples": 20, 
        "evaluate_plot_num_samples": 15,
        "plot_num_samples": 1,
        "num_batches": 2,
        "discretisation": 1,
        "encoder_channels": 32,
        "decoder_channels": 32,
        "num_layers": 6,
        "nc_bounds": [80, 100],
        "nt_bounds": [40, 50],
        ## number of training/validation/evaluation points per epoch not implemented, instead gives number of points per batch (approx. 100) * num_batches points for all three cases
    }

    if _config['evaluate']:
        _config.update(mode='_evaluate')
    else:
        _config.update(mode='_train')

    config = dict2dot(_config)

    main(config, _config)