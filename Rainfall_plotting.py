import lab as B
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import numpy as np


def rainfall_plotter(
    state: B.RandomState, 
    model, 
    generator, 
    save_path: str, 
    xbounds: Tuple[float, float], 
    ybounds: Tuple[float, float], 
    reference: bool = False,
    device: str = 'cpu',
    num_samples: int = 25,
    ):
    sns.set_theme()

    with B.on_device(device):
        epoch = generator.epoch()
        batch = epoch[0]    

    # Build grid for plotting model rainfall predictions spatially (not just context and target outputs and predictions)
    num = 200
    hmap_x1 = np.linspace(xbounds[0], xbounds[1], num)
    hmap_x2 = np.linspace(ybounds[0], ybounds[1], num)
    hmap_X = np.meshgrid(hmap_x1, hmap_x2)
    hmap_X = np.array(hmap_X).reshape((2, num**2)) # of shape (2, num, num) = (coords, xs, ys) to (2, num^2)
    yc_bernoulli, yc_precip = batch['yc']
    yc_bernoulli, yc_precip = yc_bernoulli.reshape(1, -1), yc_precip.reshape(1, -1)
    xc = batch['xc']

    with torch.no_grad():
        _, hmap_dist = model(
            state, 
            xc=xc,
            yc=batch['yc'],
            xt=B.cast(torch.float32, hmap_X),
            )
    hmap_bernoulli = B.to_numpy(hmap_dist.bernoulli_prob[0])
    hmap_kappa, hmap_chi = hmap_dist.z_gamma # (num, ), (num, ) 

    hmap_rain = np.zeros(len(hmap_bernoulli))
    for i, (kappa_i, chi_i) in zip(hmap_kappa, hmap_chi):
        sample_i = np.mean(np.random.gamma(shape=kappa_i, scale=chi_i, size=num_samples))
        ind = np.where(hmap_bernoulli==1)[i]
        hmap_rain[ind] = sample_i
    hmap_rain = hmap_rain.reshape((num, num)) # (1, num ^ 2) for 1 output dimension and _num points: (1, num ^ 2) => (num ^ 2, ) when selecting hmap_class

    # Plot the Heatmap with generated precipitation predictions over gridded inputs, with or without the true precipitation next to it for comparison
    if reference:
        _, (ax1, ax2) = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (5,10))
        ax1.pcolormesh(hmap_X[0], hmap_X[1], hmap_rain, vmin = 0, cmap = 'Pastel2', alpha = 0.4, shading = 'auto')   
        ax1.scatter(batch['xc'].to_numpy()[0], batch['xc'].to_numpy()[1], marker = 'o', color='k', label='Context Points')
        ax1.scatter(batch['xt'].to_numpy()[0], batch['xt'].to_numpy()[1], marker = '+', color='k', label='Target Points')
        ax1.colorbar()
        ax1.set_title(f'Model Predicted Rainfall')
        ax2.imshow(batch['reference'], alpha = 0.4)
        ax2.set_cmap('Pastel2')
        ax2.colorbar()
        ax2.set_title(f'True Rainfall')
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()
    else:
        plt.pcolormesh(hmap_X[0], hmap_X[1], hmap_rain, vmin = 0, cmap = 'Pastel2', alpha = 0.4, shading = 'auto')   
        plt.colorbar()
        plt.scatter(batch['xc'].to_numpy()[0], batch['xc'].to_numpy()[1], marker = 'o', color='k', label='Context Points')
        plt.scatter(batch['xt'].to_numpy()[0], batch['xt'].to_numpy()[1], marker = '+', color='k', label='Target Points')
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()