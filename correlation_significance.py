from xarray import corr
import neuralprocesses.torch as nps
import torch
import lab as B
import wbml.out as out
from wbml.experiment import WorkingDirectory
import experiment as exp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functools import partial
import matrix
from tqdm import tqdm


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def train(state, model, opt, objective, gen):
    vals = []
    for batch in tqdm(gen.epoch()):
        state, obj = objective(
            state,
            model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
        )
        vals.append(B.to_numpy(obj))
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()

    out.kv("Loglik (T)", exp.with_err(B.concat(*vals)))
    return state, [B.mean(B.concat(*vals)), 1.96 * B.std(B.concat(*vals)) / B.sqrt(B.length(B.concat(*vals)))]


def eval(state, model, objective, gen):
    with torch.no_grad():
        vals = []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
            )

            n = nps.num_data(batch["xt"], batch["yt"])
            vals.append(B.to_numpy(obj))

    out.kv("Loglik (V)", exp.with_err(B.concat(*vals)))
    return state, [B.mean(B.concat(*vals)), 1.96 * B.std(B.concat(*vals)) / B.sqrt(B.length(B.concat(*vals)))]


def build_corrconvnp(config):
    model = nps.construct_convgnp(
        likelihood = 'het',
        lv_likelihood = 'lowrank',
        dim_lv = config.dim_lv,
        dim_x=config.dim_x,
        dim_y=config.dim_y,
        points_per_unit=config.points_per_unit,
        unet_channels=config.unet_channels,
        unet_kernels=config.unet_kernels,
        conv_layers=config.num_layers,   
    )
    return model


def build_convnp(config):
    ##NOTE: the reason for this is because the 'het' lieklihood has a different number of channels to the 'lowrank' for the same shape covariance.
    ## Therefore, build 'lowrank' again and then force to be diagonal.
    corrconvnp = build_corrconvnp(config)
    model = nps.Model(
        nps.Chain(
            corrconvnp.encoder,
            lambda dist: nps.MultiOutputNormal(dist._mean, matrix.Diagonal(B.diag(dist._var)), dist._noise, dist.shape),
        ),
        corrconvnp.decoder,
    )
    return model


def run(config):

    out.report_time = True
    B.epsilon = 1e-8
    wd = WorkingDirectory(
        *config.root,
        config.data,
        *((f"x{config.dim_x}_y{config.dim_y}",) if hasattr(config, "dim_x") else ()),
        'CorrelationSignificance',
        *((config.arch,) if hasattr(config, "arch") else ()),
        config.objective,
        log=f"log{config.mode}.txt",
        diff=f"diff{config.mode}.txt",
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    B.set_global_device(device)
    state = B.create_random_state(torch.float32, seed=1)

    objective = partial(
            nps.elbo,
            num_samples=config.num_samples,
            subsume_context=True,
            normalise=True,
        )

    data_generator, _, _ = exp.data[config.data]["setup"](
        config,
        config,
        num_tasks_train=2**9 if config.train_test else 2**14,
        num_tasks_cv=2**9 if config.train_test else 2**12,
        num_tasks_eval=2**9 if config.evaluate_fast else 2**14,
        device=device,
    )

    B.epsilon = 1e-2

    model = build_corrconvnp(config)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), config.rate)
    vals_corr, vals_corr_cv = [], []
    for i in range(0, config.epochs):
        with out.Section(f"Epoch {i + 1}"):
            _, _val_corr = train(state, model, opt, objective, data_generator)
            _, _val_corr_cv = eval(state, model, objective, data_generator)
            vals_corr.append(_val_corr)
            vals_corr_cv.append(_val_corr_cv)

    torch.save(
        {
        'weights': model.encoder.state_dict(), ## save only encoder part 
        },
        wd.file(f"model-last-corrconvnp-encoder.torch"),
    )
    torch.save(
        {
        'weights': model.decoder.state_dict(), ## save only encoder part 
        },
        wd.file(f"model-last-corrconvnp-decoder.torch"),
    )

    model = build_convnp(config)
    model = model.to(device)

    model.encoder[0].load_state_dict(
        torch.load(wd.file("model-last-corrconvnp-encoder.torch"), map_location=device)["weights"]
    )
    model.decoder.load_state_dict(
        torch.load(wd.file("model-last-corrconvnp-decoder.torch"), map_location=device)["weights"]
    )

    opt = torch.optim.Adam(model.parameters(), config.rate)
    vals, vals_cv = [], []
    for i in range(config.epochs, 2 * config.epochs):
        with out.Section(f"Epoch {i + 1}"):
            _, _val = train(state, model, opt, objective, data_generator)
            _, _val_cv = eval(state, model, objective, data_generator)
            vals.append(_val)
            vals_cv.append(_val_cv)

    torch.save(
        {
            "weights": model.state_dict(),
        },
        wd.file(f"model-last-convnp.torch"),
    )

    return vals_corr, vals_corr_cv, vals, vals_cv, wd


def plot(vals_corr, vals_corr_cv, vals, vals_cv, wd):
    sns.set_theme()
    xs = np.arange(2*config.epochs)

    linear_means = -np.array([np.array(vals_corr)[:,0], np.array(vals)[:,0]]).flatten()
    linear_vars = np.array([np.array(vals_corr)[:,1], np.array(vals)[:,1]]).flatten()
    plt.plot(xs + 1, linear_means, 'xkcd:royal blue', label='Training')
    plt.fill_between(xs + 1, linear_means - linear_vars, linear_means + linear_vars, color='xkcd:royal blue', alpha=0.4)

    cv_linear_means = -np.array([np.array(vals_corr_cv)[:,0], np.array(vals_cv)[:,0]]).flatten()
    cv_linear_vars = np.array([np.array(vals_corr_cv)[:,1], np.array(vals_cv)[:,1]]).flatten()
    plt.plot(xs + 1, cv_linear_means, 'xkcd:red', label='Validation')
    plt.fill_between(xs + 1, cv_linear_means - cv_linear_vars, cv_linear_means + cv_linear_vars, color='xkcd:red', alpha=0.4)

    plt.title('Training loss with epochs')
    plt.plot(np.ones(100)*len(linear_means)//2, np.linspace(min(linear_means), max(linear_means), 100), 'k--', alpha=0.4)
    plt.xlabel('Epochs')
    plt.ylabel('Negative Log Likelihood')
    plt.legend()
    plt.savefig(wd.file()+f"/CorrChange.png")
    plt.close()


if __name__ == '__main__':

    _config = {
        'root': ['_experiments'],
        'dim_x': 1,
        'dim_y': 1,
        'dim_yc': 1,
        'dim_lv': 1, 
        'dim_yt': 1,
        'epochs': 10,
        'arch': 'unet',
        'objective': 'elbo', ## HC
        'mode': '_train', ## HC 
        'data': 'eq',    
        'batch_size': 16,
        'train_test': False,    
        'rate': 3e-4,
        'num_samples': 20,
        "width": 256,
        "dim_embedding": 256,
        "num_heads": 8,
        "num_layers": 6,
        "unet_channels": (64, 64, 64, 128, 128, 128, 256),
        "unet_kernels": 5,
        "num_basis_functions": 512,
        'fix_noise': True, 
        "points_per_unit": 64,
        }
    config = dotdict(_config)

    vals_corr, vals_corr_cv, vals, vals_cv, wd = run(config)

    plot(vals_corr, vals_corr_cv, vals, vals_cv, wd)