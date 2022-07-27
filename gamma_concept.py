import numpy as np
import lab.torch as B
import torch
from scipy.special import gamma
import neuralprocesses.torch as nps
from plum import convert
from functools import partial
from wbml.experiment import WorkingDirectory
import wbml.out as out
from torch_gamma_data_gens import gp_example
from torch_gamma_plotting import plot_1d, plot_2d
import experiment as exp
from scipy.stats import loggamma


class GammaDistribution:

    def __init__(self, params, device):
        self.kappa = params[..., 0:1, :] # shape (*b, c=2, n) -> (*b, 1, n)
        self.chi = params[..., 1:2, :] # shape (*b, c=2, n) -> (*b, 1, n)
        self.device = device

        print(f'\nKappa range: {torch.min(self.kappa.flatten()):.2f} to {torch.max(self.kappa.flatten()):.2f}')
        print(f'Chi range: {torch.min(self.chi.flatten()):.2f} to {torch.max(self.chi.flatten()):.2f}\n')

    def logpdf(self, y_amount):
        # each of shape (b, 1, n) whereas kappa, chi possibly of shape (s, b, 1, n), but broadcasting is automatic.
            
        return B.sum(
            (self.kappa - 1) * torch.nan_to_num(torch.log(y_amount), 0.) - self.chi * y_amount + self.kappa * torch.log(self.chi) - torch.log(torch.tensor(gamma(self.kappa.detach().cpu().numpy())).to(self.device)),
            axis=(-2, -1),
        )


def construct_gamma_model(
    dim_x = 1, 
    dim_y = 1, 
    dim_lv = 16,
    lv_likelihood = 'lowrank',
    discretisation = 1,
    device='cpu',
    ):

    if dim_y != 1:
        print("'dim_y' MUST equal 0 for binary classification. Defaulted to 1.")
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
    # TODO: finish making dim_lv = 0 capable
    decoder_channels = (32,) * 6
    encoder_channels = (32,) * 6
    if dim_lv > 0:
        unet_latent_variable = nps.UNet(
            dim=dim_x,
            in_channels=2 * dim_y,
            out_channels=out_channels,    # ensure correct rank - this is 'num_channels' in likelihood definition and differs depending on encoder likelihood
            channels=encoder_channels,
        )
        
        unet = nps.UNet(
            dim=dim_x,
            in_channels=dim_lv,
            out_channels=2 * dim_y, # 2 per output dimension as Gamma has two sufficient statistics
            channels=decoder_channels,
        )

    else:
        dim_yc = convert(dim_y, tuple)
        unet = nps.UNet(
            dim=dim_x,
            in_channels = sum(dim_yc) + len(dim_yc),
            out_channels=2 * dim_y, # 2 per output dimension as Gamma has two sufficient statistics
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
        lambda z: GammaDistribution(B.softplus(z), device=device),
    )
    model = nps.Model(encoder, decoder)

    return model


def train(state, model, opt, objective, gen, *, epoch):
    """Train for an epoch."""
    vals = []
    for i, batch in enumerate(gen.epoch()):
        if i == 0: print('data gen device:', B.device(batch['xc']))
        state, obj = objective(
            state,
            model,
            batch["xc"],
            batch["yc"],
            batch["xt"],
            batch["yt"],
            # epoch=epoch,
        )
        vals.append(B.to_numpy(obj))
        # Be sure to negate the output of `objective`.
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()
    
    ## changed as outputs are 1d from decoder now. Takes '*vals' as a 1D numpy array:
    vals = np.array(vals)
    out.kv("Loglik (T)", exp.with_err(B.concat(*vals)))
    return state, B.mean(B.concat(*vals))


def eval(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():
        vals = []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["xc"],
                batch["yc"],
                batch["xt"],
                batch["yt"],
            )

            # Save numbers.
            vals.append(B.to_numpy(obj))

        ## changed as outputs are 1d from decoder now. Takes '*vals' as a 1D numpy array:
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
    model = construct_gamma_model(
        dim_x = config.dim_x,
        dim_y = config.dim_y,
        dim_lv = config.dim_lv,
        lv_likelihood = config.lv_likelihood,
        discretisation = config.discretisation,
        device=device,
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

    # Tensors are always of the form `(b, c, n)`.
    if config.data not in ['gamma_gp']:
        print(f'Data generator {config.data} not implemented, defaulting to Example')
        config.data = 'gamma_gp'

    if config.data == 'gamma_gp':
        dim_x = config.dim_x
        gen_train, gen_cv, gens_eval = [    
            gp_example(dim_x=dim_x, batch_size=config.batch_size, device=device, nc_bounds=config.nc_bounds, nt_bounds=config.nt_bounds),
            gp_example(dim_x=dim_x, batch_size=config.batch_size, device=device, nc_bounds=config.nc_bounds, nt_bounds=config.nt_bounds),
            gp_example(dim_x=dim_x, batch_size=1, device=device, nc_bounds=config.nc_bounds, nt_bounds=config.nt_bounds, reference=True),
            ]

    objective = partial(
            nps.elbo,
            num_samples=config.num_samples,
            subsume_context=True,
            normalise=True,
        )
    objective_cv = partial(
            nps.elbo,
            num_samples=config.num_samples,
            subsume_context=False,  
            normalise=True,
        )
    objective_eval = partial(
            nps.elbo,
            num_samples=5,
            subsume_context=False,  
            normalise=True,
        )
        
    ## In an evaluation regime only:
    if config.evaluate:

        if config.evaluate_last:
            name = "model-last.torch"
        else:
            name = "model-best.torch"
        model.load_state_dict(torch.load(wd.file(name), map_location=device)["weights"])

        for i in range(config.evaluate_plot_num_samples):
            if config.dim_x == 1:
                plot_1d(state, model, gens_eval, wd.file()+f"/evaluate-{i + 1:03d}.png", device=device)
            elif config.dim_x == 2:
                plot_2d(state, model, gens_eval, wd.file()+f"/evaluate-{i + 1:03d}.png", device=device, xbounds=[0, 60], ybounds=[0, 60], reference=True)

        with out.Section('ELBO'):
            state, _ = eval(state, model, objective_eval, gen_cv)


    ## In a training regime:
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

                if config.dim_x == 1:
                    plot_1d(state, model, gens_eval, wd.file()+f"/train-{i + 1:03d}.png", device=device)
                elif config.dim_x == 2:
                    plot_2d(state, model, gens_eval, wd.file()+f"/train-{i + 1:03d}.png", device=device, xbounds=[0, 60], ybounds=[0, 60], reference=True)


if __name__ == '__main__':

    class dict2dot(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    _config = {
        ## Model configs are hardcoded in 'construct_bernoulli_model' - change them there
        ## below are configs that do change the nature of the model/data/working directory
        "likelihood": 'gamma',
        "arch": 'unet',
        "objective": 'elbo',
        "model": 'ConvCorrGNP',
        "dim_x": 2,
        "dim_y": 1, # NOTE: Has to be the case for binary classification
        "dim_lv": 16, # TODO: is a high LV dim detrimental?
        "data": 'gamma_gp',
        "lv_likelihood": 'lowrank',
        "root": ["_experiments"],
        "epochs": 30,
        "resume_at_epoch": None, 
        "train_test": None,
        "evaluate": False,
        "evaluate_fast": False, # NOTE: Not implemented
        "rate": 3e-4,
        "evaluate_last": False,
        "evaluate_num_samples": 1024,
        "num_samples": 20,
        "evaluate_last": False,
        "evaluate_plot_num_samples": 15,
        "plot_num_samples": 1,
        "fix_noise": True, # NOTE: Not implemented
        "batch_size": 16,
        "discretisation": 1, # NOTE: make small when dealing with large xrange (e.g. on gp-cutoff)
        "nc_bounds": [80, 100],
        "nt_bounds": [40, 50],
        ## number of training/validation/evaluation points not implemented, instead gives number of points per batch (approx. 15) * num_batches points for all three cases
    }

    if _config['evaluate']:
        _config.update(mode='_evaluate')
    else:
        _config.update(mode='_train')

    config = dict2dot(_config)

    main(config, _config)