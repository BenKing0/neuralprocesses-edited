import shutup
from sklearn.metrics import mean_poisson_deviance
import neuralprocesses.torch as nps
import torch
import lab.torch as B
import wbml.out as out
import experiment as exp
import numpy as np
from plum import convert
from functools import partial
from wbml.experiment import WorkingDirectory
from torch_classification_data_gens import example_data_gen, gp_cutoff
from torch_classification_plotting import plot_classifier_1d, plot_classifier_2d
shutup.please()

class BernoulliDistribution(torch.nn.Module):
    def __init__(self, probs):
        super().__init__()
        self.probs = probs

    def logpdf(self, y):    ## define a CLASSIFICATION probability for calculation of loglik in ELBO non lv part.
        if 0. not in self.probs and 1. not in self.probs:
            return B.sum(
                B.log(self.probs) * y + B.log(1 - self.probs) * (1 - y),
                axis=-1,
            )   ## axis either -1 or (-2, -1)
        else:
            return 0

    def class_1_prob(self):
        ## To access outside where likelihood defined in model building.
        ## Or just use 'dist(=BernoulliLikelihood).probs'
        return self.probs


def construct_bernoulli_model(  
    dim_x = 1, 
    dim_y = 1, 
    dim_lv = 16,
    lv_likelihood = 'lowrank',
    discretisation = 16,
    ):
    """
    Construct a model with a Gaussian heterogenous or a lowrank approximated encoder likelihood (if dim_lv > 0)
    and a Bernoulli decoder likelihood for binary classification. 'dim_y' MUST equal 1 for binary classification.
    """

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
    ##TODO: finish making dim_lv = 0 capable
    decoder_channels = (32,) * 6
    encoder_channels = (32,) * 6
    if dim_lv > 0:
        unet_latent_variable = nps.UNet(
            dim=dim_x,
            in_channels=2 * dim_y,
            out_channels=out_channels,    ## ensure correct rank - this is 'num_channels' in likelihood definition and differs depending on encoder likelihood
            channels=encoder_channels,
        )
        
        unet = nps.UNet(
            dim=dim_x,
            in_channels=dim_lv,
            out_channels=1 * dim_y, ## 1 per output dimension as Bernoulli has a single sufficient statistic
            channels=decoder_channels,
        )

    else:
        dim_yc = convert(dim_y, tuple)
        unet = nps.UNet(
            dim=dim_x,
            in_channels = sum(dim_yc) + len(dim_yc),
            out_channels=1 * dim_y, ## 1 per output dimension as Bernoulli has a single sufficient statistic
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
        lambda z: BernoulliDistribution(B.sigmoid(z)),  ## softplus trained way faster but caused numerical issues when self.probs > 1
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
    out.kv("Loglik (T)", exp.with_err(vals))
    return state, B.mean(vals)


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
            n = nps.num_data(batch["xt"], batch["yt"])
            vals.append(B.to_numpy(obj))

        ## changed as outputs are 1d from decoder now. Takes '*vals' as a 1D numpy array:
        vals = np.array(vals)
        out.kv("Loglik (V)", exp.with_err(vals))
        return state, B.mean(vals)


def main(config, _config):

    model = construct_bernoulli_model(
        dim_x = config.dim_x,
        dim_y = config.dim_y,
        dim_lv = config.dim_lv,
        lv_likelihood = config.lv_likelihood,
        discretisation = config.discretisation,
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    B.set_global_device(device)
    print('global device: ', B.ActiveDevice.active_name)
    print('model device: ', device)
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
    if config.data not in ['binary_MoG', 'gp_cutoff']:
        print('Data generator has to be a classification one, defaulting to Binary MoG.')
        config.data = 'binary_MoG'

    # gen_train, gen_cv, gens_eval = exp.data[config.data]["setup"](
    #     config,
    #     _config,
    #     num_tasks_train=2**9 if config.train_test else 2**14,
    #     num_tasks_cv=2**9 if config.train_test else 2**12,
    #     num_tasks_eval=2**9 if config.evaluate_fast else 2**14,
    #     device=device,
    # )

    if config.data == 'binary_MoG':
        means = [[-1] , [1]]
        covariances = [[0.5], [0.5]]
        dim_x = config.dim_x
        assert np.array(means[0]).ndim == dim_x and np.array(covariances[0]).ndim == dim_x
        priors = [0.5, 0.5]
        assert sum(priors) == 1.
        ## generated epochs from these are of shape (b, c, n)
        gen_train, gen_cv, gens_eval = [    
            example_data_gen(means, covariances, dim_x=dim_x, num_batches=config.num_batches, priors=priors, device=device),
            example_data_gen(means, covariances, dim_x=dim_x, num_batches=config.num_batches, priors=priors, device=device),
            example_data_gen(means, covariances, dim_x=dim_x, num_batches=config.num_batches, priors=priors, device=device),
            ]

    elif config.data == 'gp_cutoff':
        xrange = [[-2]*config.dim_x, [2]*config.dim_x]
        means, covariances, priors = None, None, None ## don't use these in plotting
        gen_train, gen_cv, gens_eval = [    
            gp_cutoff(config.dim_x, xrange, num_batches=config.num_batches, device=device),
            gp_cutoff(config.dim_x, xrange, num_batches=config.num_batches, device=device),
            gp_cutoff(config.dim_x, xrange, num_batches=config.num_batches, device=device),
            ]

    else:
        pass

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
                plot_classifier_1d(state, model, gens_eval, wd.file()+f"/evaluate-{i + 1:03d}.png", means=means, vars=covariances, prior=priors, device=device)
            elif config.dim_x == 2:
                plot_classifier_2d(state, model, gens_eval, wd.file()+f"/evaluate-{i + 1:03d}.png", device=device)

        with out.Section('ELBO'):
            # for gen_name, gen in gens_eval():
            #     with out.Section(gen_name.capitalize()):
            #         state, _ = eval(state, model, objective_eval, gen)
            state, _ = eval(state, model, objective_eval, gens_eval)


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
                    plot_classifier_1d(state, model, gens_eval, wd.file()+f"/train-{i + 1:03d}.png", means=means, vars=covariances, prior=priors, device=device)
                elif config.dim_x == 2:
                    plot_classifier_2d(state, model, gens_eval, wd.file()+f"/train-{i + 1:03d}.png", device=device)


if __name__ == '__main__':

    class dict2dot(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    _config = {
        ## Model configs are hardcoded in 'construct_bernoulli_model' - change them there
        ## below are configs that do change the nature of the model/data/working directory
        "likelihood": 'bernoulli',
        "arch": 'unet',
        "objective": 'elbo',
        "model": 'ConvCorrBNP',
        "dim_x": 1,
        "dim_y": 1, ##NOTE: Has to be the case for binary classification
        "dim_lv": 1,
        "data": 'gp_cutoff',   ##NOTE: Not yet implemented
        "lv_likelihood": 'lowrank',
        "root": ["_experiments"],
        "epochs": 1,
        "resume_at_epoch": None, 
        "train_test": None,
        "evaluate": False,
        "evaluate_fast": False, ##NOTE: Not implemented
        "rate": 3e-4,
        "evaluate_last": False,
        "evaluate_num_samples": 1024,   ##NOTE: What does num_samples mean for bernoulli?
        "num_samples": 20,   ##NOTE: What does num_samples mean for bernoulli?
        "evaluate_last": False,
        "evaluate_plot_num_samples": 15,
        "plot_num_samples": 1,
        "fix_noise": True, ##NOTE: Not implemented
        "num_batches": 1,
        "discretisation": 16,
        ## number of training/validation/evaluation points not implemented, instead gives number of points per batch (approx. 15) * num_batches points for all three cases
    }

    if _config['evaluate']:
        _config.update(mode='_evaluate')
    else:
        _config.update(mode='_train')

    config = dict2dot(_config)

    main(config, _config)


##TODO:
## *low priority* 3. Finish building full data generator (need?)

##NOTE: long term TODO
## - compare overfitting (i.e. steepness of sigmoid produced) with output of neural network trained on same data and see if this meta learning neuralprocess approach is better