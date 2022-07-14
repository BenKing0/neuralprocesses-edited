import lab as B
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import numpy as np
from tqdm import tqdm


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
            [(xc, yc_bernoulli), (xc, yc_precip)],
            B.cast(torch.float32, hmap_X),
            )
    hmap_bernoulli = B.to_numpy(hmap_dist.bernoulli_prob[0, 0])
    hmap_kappa, hmap_chi = hmap_dist.z_gamma[0] # (num, ), (num, ) 

    # TODO: must be faster way to do this
    hmap_rain = np.zeros(len(hmap_bernoulli))
    for i, (kappa_i, chi_i, bern_i) in enumerate(zip(hmap_kappa, hmap_chi, hmap_bernoulli)):
        if bern_i > 0.5:
            # TODO: should this be a mean over samples?
            sample_i = np.mean(np.random.gamma(shape=kappa_i.detach().cpu().numpy(), scale=chi_i.detach().cpu().numpy(), size=num_samples))
            hmap_rain[i] = sample_i
        else:
            hmap_rain[i] = 0
    hmap_rain = hmap_rain.reshape((num, num)) # (1, num ^ 2) for 1 output dimension and _num points: (1, num ^ 2) => (num ^ 2, ) when selecting hmap_class

    # Plot the Heatmap with generated precipitation predictions over gridded inputs, with or without the true precipitation next to it for comparison
    if reference:
        _, (ax1, ax2) = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (5,10))
        plot1 = ax1.imshow(hmap_rain, cmap='Pastel2', alpha=0.5, vmin=0, vmax=np.max(B.to_numpy(batch['reference'])), extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]])
        ax1.scatter(B.to_numpy(batch['xc'])[0], B.to_numpy(batch['xc'])[1], marker = 'o', color='k', label='Context Points', s=0.1)
        # ax1.scatter(B.to_numpy(batch['xt'])[0], B.to_numpy(batch['xt'])[1], marker = '+', color='k', label='Target Points', s=0.1)
        ax1.set_title(f'Model Predicted Rainfall')
        # ax1.legend()
        plot2 = ax2.imshow(batch['reference'], alpha = 0.5, cmap='Pastel2', vmin=0, vmax=np.max(B.to_numpy(batch['reference'])), extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]])
        ax2.set_title(f'True Rainfall')
        plt.colorbar(plot1, ax=ax1, shrink=0.2)
        plt.colorbar(plot2, ax=ax2, shrink=0.2)
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()
    else:
        plt.imshow(hmap_rain, cmap='Pastel2', alpha=0.5, vmin=0)
        plt.scatter(B.to_numpy(batch['xc'])[0], B.to_numpy(batch['xc'])[1], marker = 'o', color='k', label='Context Points', s=0.1)
        # plt.scatter(B.to_numpy(batch['xt'])[0], B.to_numpy(batch['xt'])[1], marker = '+', color='k', label='Target Points', s=0.1)
        plt.colorbar()
        # plt.legend()
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()