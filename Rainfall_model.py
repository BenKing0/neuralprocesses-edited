from ctypes import sizeof
import re
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
from Rainfall_data import rainfall_generator, Bernoulli_Gamma_synthetic
from Rainfall_plotting import rainfall_plotter
from scipy.special import gamma
from typing import List
from neuralprocesses.model.elbo import _merge_context_target
from sys import getsizeof
shutup.please()


# `Aggregate` outputs are assumed to always come with `AggregateInput` inputs.
# In this case, that's not true, so add support for this.

@_merge_context_target.dispatch
def _merge_context_target(contexts: List, xt: B.Numeric, yt: base.Aggregate):
    return _merge_context_target(
        contexts,
        nps.AggregateInput(*((xt, i) for i in range(len(yt)))),
        yt,
    )


# NOTE: all functions built for passing in a single batch batch_size times each epoch


class BernoulliDistribution:

    def __init__(self, probs):
        self.probs = probs[0, :] # (c=1, n) -> (n, )

    def logpdf(self, y): 
        # y is shape (n, )

        return B.sum(
            B.log(self.probs) * y + B.log(1 - self.probs) * (1 - y),
            axis=0,
        )
                

class GammaDistribution:

    def __init__(self, kappa, chi):
        self.kappa = kappa[0, :] # shape (c=1, n_adjusted) -> (n_adjusted, )
        self.chi = chi[0, :] # shape (c=1, n_adjusted) -> (n_adjusted, )

    def logpdf(self, y):
        # y is of shape(n_adjusted, )
            
        return B.sum(
            (self.kappa - 1) * B.log(y) - (y / self.chi) - (B.log(gamma(self.kappa)) + self.kappa * B.log(self.chi)),
            axis=0,
        )


class BernoulliGammaDist:

    def __init__(self, z_bernoulli, z_gamma):

        if len(z_bernoulli.shape) == 2 and len(z_gamma.shape) == 2:
            z_bernoulli, z_gamma = z_bernoulli.reshape(1, *z_bernoulli.shape), z_gamma.reshape(1, *z_gamma.shape)

        # These are the model predictions for bernoulli statistic and gamma statistics.
        self.bernoulli_prob = B.sigmoid(z_bernoulli) # shape (samples, c=1, n)
        self.z_gamma = B.softplus(z_gamma) # shape (samples, c=2, n)

    def logpdf(self, ys):

        y_rain, y_amount = ys
        y_rain = y_rain[0] # (1, n) -> (n,)
        y_amount = y_amount[0] # (1, n) -> (n,) 
        samples_loglik = []

        for s in range(self.z_gamma.shape[0]):

            bernoulli_dist = BernoulliDistribution(self.bernoulli_prob[s])

            y_amount_adjusted = [] # adjusted to only include days that it did rain. 
            z_gamma_adjusted = [] # only look at model's predicted gamma statistics if y says it did rain. 
            for i, [rain, amount] in enumerate(zip(y_rain, y_amount)):

                if rain:
                    y_amount_adjusted.append(amount.cpu().detach().numpy())
                    z_gamma_adjusted.append(self.z_gamma[s, :, i].cpu().detach().numpy())
            
            if y_amount_adjusted:
                y_amount_adjusted = torch.tensor(np.array(y_amount_adjusted), dtype=torch.float32) # Should be size (n_adjusted, )
                z_gamma_adjusted = torch.tensor(np.array(z_gamma_adjusted).T, dtype=torch.float32) # Should be size (c=2, n_adjusted)
                gamma_dist = GammaDistribution(z_gamma_adjusted[0, :].reshape(1, -1), z_gamma_adjusted[1, :].reshape(1, -1))
                samples_loglik.append(bernoulli_dist.logpdf(y_rain) + gamma_dist.logpdf(y_amount_adjusted)) # sum should be shape (samples, )

            else:
                samples_loglik.append(bernoulli_dist.logpdf(y_rain)) # should be shape (samples, )

        return torch.tensor(samples_loglik, dtype=torch.float32) # should be shape (samples, )


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
        ):

        if arch == 'unet':
            decoder_channels = (encoder_num_channels,) * num_layers
            encoder_channels = (decoder_num_channels,) * num_layers

            net_latent_variable = nps.UNet(
                dim=2,
                in_channels=4,
                out_channels=dim_lv * (2 + 64),
                channels=encoder_channels,
                separable=True,
            )
            
            net = nps.UNet(
                dim=2,
                in_channels=dim_lv,
                out_channels=3,
                channels=decoder_channels,
                separable=True,
            )

        elif arch == 'convnet':

            decoder_channels = encoder_num_channels
            encoder_channels = decoder_num_channels

            net_latent_variable = nps.ConvNet(
                dim=2,
                in_channels=4,
                out_channels=dim_lv * (2 + 64),
                channels=encoder_channels,
                separable=True,
                num_layers=num_layers,
                points_per_unit=discretisation,
                receptive_field=4,
            )
            
            net = nps.ConvNet(
                dim=2,
                in_channels=dim_lv,
                out_channels=3,
                channels=decoder_channels,
                separable=True,
                num_layers=num_layers,
                points_per_unit=discretisation,
                receptive_field=4,
            )

        model = nps.Model(
            nps.FunctionalCoder(
                nps.Discretisation(
                    points_per_unit=discretisation,
                    multiple=1,
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
                nps.SetConv(scale=2 / discretisation),
                nps.Splitter(1, 2),
                lambda xs: BernoulliGammaDist(*xs),
            ),
        )

        return model


class separate:

    @classmethod
    def _bernoulli_model( 
        cls, 
        dim_x = 2, 
        dim_lv = 16,
        lv_likelihood = 'lowrank',
        discretisation = 1,
        ):
        """
        Bernoulli likelihood part of the model which decides whether it is raining or not (binary).
        """
        dim_y = 1

        if dim_lv > 0:
            if lv_likelihood == 'het':
                encoder_likelihood = nps.HeterogeneousGaussianLikelihood()
                out_channels = 2 * dim_lv
            elif lv_likelihood == 'lowrank':
                encoder_likelihood = nps.LowRankGaussianLikelihood(64)
                out_channels = (2 + 64) * dim_lv
            else:
                print(f'Non-implemented likelihood passed ({lv_likelihood} is not in "het" or "lowrank"), defaulting to "lowrank" for Correlated LV model.')
                encoder_likelihood = nps.LowRankGaussianLikelihood(64)
                out_channels = (2 + 64) * dim_lv
        else:
            encoder_likelihood = nps.DeterministicLikelihood()
            unet_latent_variable = lambda x: x


        # CNN architecture:
        ##TODO: finish making dim_lv = 0 capable
        decoder_channels = (32,) * 6
        encoder_channels = (32,) * 6
        if dim_lv > 0:
            unet_latent_variable = nps.UNet(
                dim=dim_x,
                in_channels=2,
                out_channels=out_channels,    ## ensure correct rank - this is 'num_channels' in likelihood definition and differs depending on encoder likelihood
                channels=encoder_channels,
            )
            
            unet = nps.UNet(
                dim=dim_x,
                in_channels=dim_lv,
                out_channels=1, ## 1 per output dimension as Bernoulli has a single sufficient statistic
                channels=decoder_channels,
            )

        else:
            dim_yc = convert(dim_y, tuple)
            unet = nps.UNet(
                dim=dim_x,
                in_channels = sum(dim_yc) + len(dim_yc),
                out_channels=1, ## 1 per output dimension as Bernoulli has a single sufficient statistic
                channels=decoder_channels,
            )

        # Discretisation of the functional embedding:
        disc = nps.Discretisation(
            points_per_unit=discretisation,
            multiple=2**unet.num_halving_layers,
            margin=0.1,
            dim=dim_x,
        )

        # Create the encoder and decoder and construct the model.
        encoder = nps.FunctionalCoder(
            disc,
            nps.Chain(
                nps.PrependDensityChannel(),
                nps.SetConv(scale=1 / disc.points_per_unit),
                nps.DivideByFirstChannel(),
                unet_latent_variable,
                encoder_likelihood,
            ),
        )
        decoder = nps.Chain(
            unet,
            nps.SetConv(scale=1 / disc.points_per_unit),
            lambda z: BernoulliDistribution(B.sigmoid(z)), 
        )
        model = nps.Model(encoder, decoder)

        return model

    @classmethod
    def _gamma_model(
        cls,
        dim_x = 2, 
        dim_lv = 16,
        lv_likelihood = 'lowrank',
        discretisation = 16,
        ):
        '''
        Gamma model which takes in 2d coordinates and returns the shape and scale parameters from a Gamma distribution, kappa and lambda, respectively.
        Inputs only passed to this model if the Bernoulli model output is rho > 0.5.
        '''
        dim_y = 2 # this is kappa and lambda from gamma distribution

        if dim_lv > 0:
            if lv_likelihood == 'het':
                encoder_likelihood = nps.HeterogeneousGaussianLikelihood()
                out_channels = 2 * dim_lv
            elif lv_likelihood == 'lowrank':
                encoder_likelihood = nps.LowRankGaussianLikelihood(64)
                out_channels = (2 + 64) * dim_lv
            else:
                print(f'Non-implemented likelihood passed ({lv_likelihood} is not in "het" or "lowrank"), defaulting to "lowrank" for Correlated LV model.')
                encoder_likelihood = nps.LowRankGaussianLikelihood(64)
                out_channels = (2 + 64) * dim_lv
        else:
            encoder_likelihood = nps.DeterministicLikelihood()
            unet_latent_variable = lambda x: x


        # CNN architecture:
        ##TODO: finish making dim_lv = 0 capable
        decoder_channels = (32,) * 6
        encoder_channels = (32,) * 6
        if dim_lv > 0:
            unet_latent_variable = nps.UNet(
                dim=dim_x,
                in_channels=2,
                out_channels=out_channels,    ## ensure correct rank - this is 'num_channels' in likelihood definition and differs depending on encoder likelihood
                channels=encoder_channels,
            )
            
            unet = nps.UNet(
                dim=dim_x,
                in_channels=dim_lv,
                out_channels=2, ## 2 per output dimension as Gamma has two sufficient statistics
                channels=decoder_channels,
            )

        else:
            dim_yc = convert(dim_y, tuple)
            unet = nps.UNet(
                dim=dim_x,
                in_channels = sum(dim_yc) + len(dim_yc),
                out_channels=2, ## 2 per output dimension as Gamma has two sufficient statistics
                channels=decoder_channels,
            )

        # Discretisation of the functional embedding:
        disc = nps.Discretisation(
            points_per_unit=discretisation,
            multiple=2**unet.num_halving_layers,
            margin=0.1,
            dim=dim_x,
        )

        # Create the encoder and decoder and construct the model.
        encoder = nps.FunctionalCoder(
            disc,
            nps.Chain(
                nps.PrependDensityChannel(),
                nps.SetConv(scale=1 / disc.points_per_unit),
                nps.DivideByFirstChannel(),
                unet_latent_variable,
                encoder_likelihood,
            ),
        )
        decoder = nps.Chain(
            unet,
            nps.SetConv(scale=1 / disc.points_per_unit),
            lambda z: GammaDistribution(B.relu(z[0]), B.relu(z[1])), # NOTE: (2, n) shape output from nps.SetConv()?
        )

        model = nps.Model(encoder, decoder)
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
        xc = batch['xc']
        yc_bernoulli, yc_precip = batch['yc']
        yt_bernoulli, yt_precip = batch['yt']
        # all var (inc. ys) must be of shape (c, n) even if c = 1
        yc_bernoulli, yc_precip, yt_bernoulli, yt_precip = yc_bernoulli.reshape(1, -1), yc_precip.reshape(1, -1), yt_bernoulli.reshape(1, -1), yt_precip.reshape(1, -1)
        obj = objective(
                model,
                [(xc, yc_bernoulli), (xc, yc_precip)],
                batch["xt"],
                nps.Aggregate(yt_bernoulli, yt_precip),
            )
        obj = obj.reshape(1, -1)
        vals.append(B.to_numpy(obj))
        # Be sure to negate the output of `objective`.
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()
    
    ## changed as outputs are 1d from decoder now. Takes 'vals' as a 1D numpy array:
    vals = np.array(vals)
    out.kv("Loglik (T)", exp.with_err(vals))
    return state, B.mean(vals)


def eval(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():
        vals = []
        batches = gen.epoch()
        for batch in batches:
            xc = batch['xc']
            yc_bernoulli, yc_precip = batch['yc']
            yt_bernoulli, yt_precip = batch['yt']
            # all var (inc. ys) must be of shape (c, n) even if c = 1
            yc_bernoulli, yc_precip, yt_bernoulli, yt_precip = yc_bernoulli.reshape(1, -1), yc_precip.reshape(1, -1), yt_bernoulli.reshape(1, -1), yt_precip.reshape(1, -1)
            obj = objective(
                model,
                [(xc, yc_bernoulli), (xc, yc_precip)],
                batch["xt"],
                nps.Aggregate(yt_bernoulli, yt_precip),
            )
            # Save numbers.
            vals.append(B.to_numpy(obj))

        ## changed as outputs are 1d from decoder now. Takes '*vals' as a 1D numpy array:
        vals = np.array(vals)
        out.kv("Loglik (V)", exp.with_err(vals))
        return state, B.mean(vals)


def main(config, _config):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    B.set_global_device(device)
    print('global device: ', B.ActiveDevice.active_name)
    print('model device: ', device)

    nc_bounds = [50, 70]
    nt_bounds = [100, 140]
    batch_size = config.num_batches

    if config.data == 'rainfall':               
            gen_train, gen_cv, gens_eval = [    
                rainfall_generator(batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, include_binary=True, device=device),
                rainfall_generator(batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, include_binary=True,  device=device),
                rainfall_generator(batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, include_binary=True,  device=device),
                ]
    else:
        gen_train, gen_cv, gens_eval = [    
                Bernoulli_Gamma_synthetic(xrange=[0, 60], batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device, kernel='eq', l=0.2, gp_mean=1, num_ref_points=30, include_binary=True),
                Bernoulli_Gamma_synthetic(xrange=[0, 60], batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device, kernel='eq', l=0.2, gp_mean=1, num_ref_points=30, include_binary=True),
                Bernoulli_Gamma_synthetic(xrange=[0, 60], batch_size=batch_size, nc_bounds=nc_bounds, nt_bounds=nt_bounds, device=device, kernel='eq', l=0.2, gp_mean=1, num_ref_points=30, include_binary=True),
            ]

    if config.type == "combined":
        model = combined.combined_model(discretisation=1,
        arch=config.arch, 
        encoder_num_channels=config.encoder_channels,
        decoder_num_channels=config.decoder_channels,
        num_layers=config.num_layers,
        dim_lv=config.dim_lv,
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
            state, _ = eval(state, model, objective_eval, gens_eval)

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
        "arch": 'convnet',
        "objective": 'elbo',
        "model": 'Rainfall',
        "dim_x": 2, # NOTE: Hard-coded, included for filename (Has to be the case for rainfall case)
        "dim_y": 1, # NOTE: Hard-coded, included for filename (Has to be the case for rainfall case)
        "dim_lv": 1, # TODO: is high LV dim detramental?
        "data": 'synthetic',
        "lv_likelihood": 'lowrank',
        "root": ["_experiments"],
        "epochs": 10,
        "train_test": None,
        "evaluate": False,
        "rate": 1e-3, # 3e-4,
        "evaluate_last": False,
        "evaluate_num_samples": 20,
        "num_samples": 20, 
        "evaluate_plot_num_samples": 15,
        "plot_num_samples": 1,
        "num_batches": 16,
        "discretisation": 2,
        "encoder_channels": 64,
        "decoder_channels": 32,
        "num_layers": 6,
        "dim_lv": 16,
        ## number of training/validation/evaluation points per epoch not implemented, instead gives number of points per batch (approx. 100) * num_batches points for all three cases
    }

    if _config['evaluate']:
        _config.update(mode='_evaluate')
    else:
        _config.update(mode='_train')

    config = dict2dot(_config)

    main(config, _config)