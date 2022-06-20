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
    for i in range(0, config.epochs):
        with out.Section(f"Epoch {i + 1}"):
            _, vals_corr = train(state, model, opt, objective, data_generator)
            _, vals_corr_cv = eval(state, model, objective, data_generator)
    torch.save(
        {
            "weights": model.state_dict(),
        },
        wd.file(f"model-last-corrconvnp.torch"),
    )

    model = build_convnp(config)
    model = model.to(device)
    model.load_state_dict(
        torch.load(wd.file("model-last-corrconv.torch"), map_location=device)["weights"]
    )
    opt = torch.optim.Adam(model.parameters(), config.rate)
    for i in range(config.epochs, 2 * config.epochs):
        with out.Section(f"Epoch {i + 1}"):
            _, vals = train(state, model, opt, objective, data_generator)
            _, vals_cv = eval(state, model, objective, data_generator)
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

    linear_means = -np.array([vals_corr[0], vals[0]]).flatten()
    linear_vars = np.array([vals_corr[1], vals[1]]).flatten()
    plt.plot(xs, linear_means, 'xkcd:royal blue')
    plt.fill_between(xs, linear_means - linear_vars, linear_means + linear_vars, 'xkcd:royal blue', alpha=0.4)

    cv_linear_means = -np.array([vals_corr_cv[0], vals_cv[0]]).flatten()
    cv_linear_vars = np.array([vals_corr_cv[1], vals_cv[1]]).flatten()
    plt.plot(xs, cv_linear_means, 'xkcd:red')
    plt.fill_between(xs, cv_linear_means - cv_linear_vars, cv_linear_means + cv_linear_vars, 'xkcd:red', alpha=0.4)

    plt.title('Training loss with epochs')
    plt.plot(np.ones(100)*len(linear_means)//2, np.linspace(min(linear_means), max(linear_means), 100), 'k--', alpha=0.4)
    plt.xlabel('Epochs')
    plt.ylabel('Negative Log Likelihood')
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
        'epochs': 1,
        'arch': 'unet',
        'objective': 'elbo', ## HC
        'mode': '_train', ## HC 
        'data': 'eq',    
        'batch_size': 1,
        'train_test': True,    
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



# def _build_adapted_convnp(dim_lv, dim_x, dim_y, dim_yc, dim_yt, unet_channels, unet_kernels, points_per_unit):

#     dim_yc, dim_yt, conv_in_channels = nps.architectures.convgnp._convgnp_init_dims(dim_yc, dim_yt, dim_y)
#     num_basis_functions = 64

#     lv_likelihood_in_channels, _, lv_likelihood = nps.architectures.util.construct_likelihood(
#         nps,
#         spec='lowrank',
#         dim_y=dim_lv,
#         num_basis_functions=num_basis_functions,
#         dtype=None,
#     )
#     encoder_likelihood = lv_likelihood

#     likelihood_in_channels, selector, likelihood = nps.architectures.util.construct_likelihood(
#         nps,
#         spec='het',
#         dim_y=dim_yt,
#         num_basis_functions=num_basis_functions,
#         dtype=None,
#     )

#     conv_out_channels = nps.architectures.convgnp._convgnp_resolve_architecture(
#         'unet',
#         unet_channels,
#         conv_channels=64,
#         conv_receptive_field=None,
#     )

#     if conv_out_channels < likelihood_in_channels:
#         linear_after_set_conv = nps.Linear(
#             in_channels=conv_out_channels,
#             out_channels=likelihood_in_channels,
#             dtype=None,
#         )
#     else:
#         conv_out_channels = likelihood_in_channels
#         linear_after_set_conv = lambda x: x

#     lv_in_channels = conv_in_channels
#     lv_out_channels = lv_likelihood_in_channels
#     in_channels = dim_lv
#     out_channels = conv_out_channels  
#     lv_conv = nps.UNet(
#         dim=dim_x,
#         in_channels=lv_in_channels,
#         out_channels=lv_out_channels,
#         channels=unet_channels,
#         kernels=unet_kernels,
#         strides=2,
#         activations=None,
#         resize_convs=False,
#         resize_conv_interp_method='nearest',
#         dtype=None,
#     )

#     conv = nps.UNet(
#         dim=dim_x,
#         in_channels=in_channels,
#         out_channels=out_channels,
#         channels=unet_channels,
#         kernels=unet_kernels,
#         strides=2,
#         activations=None,
#         resize_convs=False,
#         resize_conv_interp_method='nearest',
#         dtype=None,
#     )
#     receptive_field = conv.receptive_field / 64

#     disc = nps.Discretisation(
#         points_per_unit=points_per_unit,
#         multiple=2**conv.num_halving_layers,
#         margin=0.1,
#         dim=dim_x,
#     )

#     model = nps.Model(
#         nps.FunctionalCoder(
#             disc,
#             nps.Chain(
#                 nps.PrependDensityChannel(),
#                 nps.architectures.convgnp._convgnp_construct_encoder_setconvs(
#                     nps,
#                     None,
#                     dim_yc,
#                     disc,
#                     None,
#                 ),
#                 nps.architectures.convgnp._convgnp_optional_division_by_density(nps, True, 1e-4),
#                 nps.Materialise(),
#                 lv_conv,
#                 encoder_likelihood, 
#                 _encoder_diag_transform, ##TODO: force this to be diagonal - to be done in coding stage when model called? Or with new part to chain? Or in model.__call__?
#             ),
#         ),
#         nps.Chain(
#             conv,
#             nps.RepeatForAggregateInputs(
#                 nps.Chain(
#                     nps.architectures.convgnp._convgnp_construct_decoder_setconv(nps, None, disc, None),
#                     linear_after_set_conv,
#                     selector,  # Select the right target output.
#                 )
#             ),
#             likelihood,
#             nps.architectures.util.parse_transform(nps, transform=None),
#         ),
#     )

#     out.kv("Receptive field", receptive_field)
#     model.receptive_field = receptive_field

#     return model