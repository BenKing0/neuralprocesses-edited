import torch
import neuralprocesses.torch as nps
import neuralprocesses as base
from neuralprocesses.model.elbo import _merge_context_target
import lab.torch as B
from plum import dispatch
from scipy.special import gamma
import numpy as np


# NOTE: built for passing in a single batch:

class BernoulliDistribution:
    def __init__(self, probs):
        self.probs = probs

    def logpdf(self, y):    
        if 0. not in self.probs and 1. not in self.probs:
            return B.sum(
                B.log(self.probs) * y + B.log(1 - self.probs) * (1 - y),
                axis=-1,
            )
        else:
            return 0


class GammaDistribution:
    def __init__(self, kappa, chi):
        self.kappa = kappa
        self.chi = chi

    def logpdf(self, y):
        return B.sum(
            (self.kappa - 1) * B.log(y) - (y / self.chi) - (B.log(gamma(self.kappa)) + self.kappa * B.log(self.chi)),
            axis = -1,
        )


class BernoulliGammaDist:
    def __init__(self, z_bernoulli, z_gamma):
        self.bernoulli_prob = B.sigmoid(z_bernoulli)[0, 0] 
        self.z_gamma = B.softplus(z_gamma[0])

    def logpdf(self, ys):

        y_bernoulli, y_gamma = ys
        y_bernoulli = y_bernoulli[0]
        y_gamma = y_gamma[0]
        bernoulli_dist = BernoulliDistribution(self.bernoulli_prob)

        y_gamma_adjusted = []
        z_gamma_adjusted = []
        for i, [rain, amount] in enumerate(zip(y_bernoulli, y_gamma)):
            if rain:
                y_gamma_adjusted.append(amount)
                z_gamma_adjusted.append(self.z_gamma[:, i].detach().numpy())

        if y_gamma_adjusted:
            y_gamma_adjusted = torch.tensor(np.array(y_gamma_adjusted), dtype=torch.float32)
            z_gamma_adjusted = torch.tensor(np.array(z_gamma_adjusted), dtype=torch.float32)
            gamma_dist = GammaDistribution(z_gamma_adjusted[:, 0], z_gamma_adjusted[:, 1])

            return bernoulli_dist.logpdf(y_bernoulli) + gamma_dist.logpdf(y_gamma_adjusted)

        else:
            return bernoulli_dist.logpdf(y_bernoulli)


model = nps.Model(
    nps.FunctionalCoder(
        nps.Discretisation(
            points_per_unit=32,
            multiple=1,
            margin=0.1,
        ),
        nps.Chain(
            nps.PrependDensityChannel(),
            nps.Parallel(
                nps.SetConv(scale=2 / 32),
                nps.SetConv(scale=2 / 32),
            ),
            nps.DivideByFirstChannel(),
            nps.Concatenate(),
            nps.ConvNet(
                dim=2,
                in_channels=4,
                out_channels=4 * (2 + 64),
                channels=32,
                num_layers=6,
                points_per_unit=32,
                receptive_field=4,
                separable=True,
            ),
            nps.LowRankGaussianLikelihood(64),
        ),
    ),
    nps.Chain(
        nps.ConvNet(
            dim=2,
            in_channels=4,
            out_channels=3,
            channels=32,
            num_layers=6,
            points_per_unit=32,
            receptive_field=4,
            separable=True,
        ),
        nps.SetConv(scale=2 / 32),
        nps.Splitter(1, 2),
        lambda xs: BernoulliGammaDist(*xs),
    ),
)

# `Aggregate` outputs are assumed to always come with `AggregateInput` inputs.
# In this case, that's not true, so add support for this.

@_merge_context_target.dispatch
def _merge_context_target(contexts: list, xt: B.Numeric, yt: base.Aggregate):
    return _merge_context_target(
        contexts,
        nps.AggregateInput(*((xt, i) for i in range(len(yt)))),
        yt,
    )


xc = B.randn(torch.float32, 2, 10)
yc_precip = B.exp(B.randn(torch.float32, 1, 10))
yc_bernoulli = B.cast(torch.float32, B.randn(torch.float32, 1, 10) > 0)

xt = B.randn(torch.float32, 2, 5)
yt_precip = B.exp(B.randn(torch.float32, 1, 5))
yt_bernoulli = B.cast(torch.float32, B.randn(torch.float32, 1, 5) > 0)

print(xc.shape, yc_bernoulli.shape, yc_precip.shape)

nps.elbo(
    model,
    [(xc, yc_bernoulli), (xc, yc_precip)],
    xt,
    nps.Aggregate(yt_bernoulli, yt_precip),
)