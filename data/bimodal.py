# from tkinter import Y
# import lab as B
# from neuralprocesses.data import SyntheticGenerator, new_batch
# from neuralprocesses.dist import UniformContinuous
# import numpy as np
# import torch
# from neuralprocesses import _dispatch

# __all__ = ["BiModalGenerator"]


# class BiModalGenerator(SyntheticGenerator):
#     """Bi-modal distribution generator.

#     Further takes in arguments and keyword arguments from the constructor of
#     :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
#     :class:`.data.SyntheticGenerator`.
#     """

#     def __init__(self, *args, **kw_args):
#         super().__init__(*args, **kw_args)


#     def generate_batch(self):
#         with B.on_device(self.device):
#             set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)
#             x = B.concat(xc, xt, axis=1)

#             phase = B.rand() * np.pi
#             amplitude = 1 + B.rand() * np.pi
#             period = np.pi + np.pi * B.rand() 

#             f = lambda x: B.cast(torch.float64, amplitude * np.sin(phase + (2*np.pi/period) * x))
#             _y = f + self._labs_bimodal(x=xc, noise=B.sqrt(self.noise))
#             y = _y(x)

#             batch = {}
#             set_batch(batch, y[:, :, :nc], y[:, :, nc:], transpose=False)
#             return batch


#     @_dispatch
#     def _labs_bimodal(self, x, state, noise, coeffs: list = [0.5, 0.5], means: list = [-0.02, 0.02]):

#         assert sum(coeffs) == 1., 'Mixture coefficients must sum to 1.'
#         torch_random_vals = noise * coeffs[0] * (B.randn(state, B.dtype(x), *B.shape(x)) + means[0]) + \
#                              noise * coeffs[1] * (B.randn(state, B.dtype(x), *B.shape(x)) + means[1])

#         return torch_random_vals


#     @_dispatch
#     def _labs_bimodal(self, x, noise, coeffs: list = [0.5, 0.5], means: list = [-0.02, 0.02]):
#         state = B.create_random_state(seed=0)
#         return self._labs_bimodal(x, state, noise, coeffs, means)


#     @_dispatch
#     def _labs_bimodal(self, x, coeffs: list = [0.5, 0.5], means: list = [-0.02, 0.02]):
#         state = B.create_random_state(seed=0)
#         noise = B.randn()
#         return self._labs_bimodal(x, state, noise, coeffs, means)


#     ## No longer in use.
#     def _numpy_bimodal(self, noise, coeffs: list = [0.5, 0.5], means: list = [-0.02, 0.02]):
        
#         assert sum(coeffs) == 1., 'Mixture coefficients must sum to 1.'
#         numpy_random_vals = coeffs[0] * np.random.normal(loc=means[0], scale=noise) + coeffs[1] * np.random.normal(loc=means[1], scale=noise)

#         return numpy_random_vals


import lab as B
import numpy as np
from lab.shape import Dimension

from neuralprocesses.data import SyntheticGenerator, new_batch

__all__ = ["BiModalGenerator"]


class BiModalGenerator(SyntheticGenerator):
    """Bi-modal distribution generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.
    """

    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)

    def __repr__(self):
        return "bimodal_generator_class"

    def generate_batch(self, variance: float = 0.2, return_ground_truth=False):
        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)
            x = B.concat(xc, xt, axis=1)

            # Draw a different random phase, amplitude, and period for every task in
            # the batch.
            self.state, rand = B.rand(
                self.state,
                self.float64,
                3,
                self.batch_size,
                1,  # Broadcast over `n`.
                1,  # There is only one input dimension.
            )
            phase = 2 * B.pi * rand[0]
            amplitude = 1 + rand[1]
            period = 1 + rand[2]

            # Construct the noiseless function.
            f = amplitude * B.sin(phase + (2 * B.pi / period) * x)

            # Add noise with variance.
            probs = B.cast(self.float64, np.array([0.5, 0.5]))
            means = B.cast(self.float64, np.array([-0.1, 0.1]))
            # Randomly choose from `means` with probabilities `probs`.
            self.state, mean = B.choice(self.state, means, self.batch_size, p=probs)
            self.state, randn = B.randn(
                self.state,
                self.float64,
                self.batch_size,
                # `nc` and `nt` are tensors rather than plain integers. Tell dispatch
                # that they can be interpreted as dimensions of a shape.
                Dimension(nc + nt),
                1,
            )
            noise = B.sqrt(variance) * randn + mean[:, None, None]

            # Construct the noisy function.
            y = f + noise
            y_ref = f

            batch = {}
            ref = {}
            set_batch(batch, y[:, :nc], y[:, nc:])
            set_batch(ref, y_ref[:, :nc], y_ref[:, nc:])

            if return_ground_truth:
                return batch, ref
            else:
                return batch