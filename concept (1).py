import neuralprocesses.torch as nps
import torch
import lab.torch as B


class BernoulliDistribution(torch.nn.Module):
    def __init__(self, probs):
        super().__init__()
        self.probs = probs

    def logpdf(self, x):
        return B.sum(
            B.log(self.probs) * x + B.log(1 - self.probs) * (1 - x),
            axis=(-2, -1),
        )


dim_x = 1
dim_y = 1
dim_lv = 16

# CNN architecture:
unet_latent_variable = nps.UNet(
    dim=dim_x,
    in_channels=2 * dim_y,
    out_channels=(2 + 64) * dim_lv,
    channels=(32,) * 6,
)
unet = nps.UNet(
    dim=dim_x,
    in_channels=dim_lv,
    out_channels=1 * dim_y,
    channels=(32,) * 6,
)

# Discretisation of the functional embedding:
disc = nps.Discretisation(
    points_per_unit=64,
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
        nps.LowRankGaussianLikelihood(64),
    ),
)
decoder = nps.Chain(
    unet,
    nps.SetConv(scale=1 / disc.points_per_unit),
    lambda z: BernoulliDistribution(B.softplus(z)),
)
convcnp = nps.Model(encoder, decoder)


# Tensors are always of the form `(b, c, n)`.
xc = B.randn(torch.float32, 16, 1, 10)                             # Context inputs
yc = B.cast(torch.float32, B.randn(torch.float32, 16, 1, 10) > 0)  # Context outputs
xt = B.randn(torch.float32, 16, 1, 15)                             # Target inputs
yt = B.cast(torch.float32, B.randn(torch.float32, 16, 1, 15) > 0)  # Target output


elbos = nps.elbo(convcnp, xc, yc, xt, yt, num_samples=1)

print(elbos)
