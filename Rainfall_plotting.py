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

    # NOTE: batch has shape (b, c, n) for each dict value
    with B.on_device(device):
        epoch = generator.epoch()
        batch = epoch[0]    

    # Build grid for plotting model rainfall predictions spatially (not just context and target outputs and predictions)
    num = 200
    hmap_x1 = np.linspace(xbounds[0], xbounds[1], num)
    hmap_x2 = np.linspace(ybounds[0], ybounds[1], num)
    hmap_X = np.meshgrid(hmap_x1, hmap_x2)
    hmap_X = np.array(hmap_X).reshape((1, 2, num**2)) # (2, num, num) = (coords, xs, ys) to (1, 2, num^2)
    yc_bernoulli, yc_precip = batch['yc'][0:1, 0:1, :], batch['yc'][0:1, 1:2, :] # (b, 2, n) to (1, 1, n) each
    yc_bernoulli, yc_precip = yc_bernoulli.reshape(1, 1, -1), yc_precip.reshape(1, 1, -1)
    xc = batch['xc'][0:1] # (b, 2, n) to (1, 2, n)

    with torch.no_grad():
        _, hmap_dist = model(
            state, 
            [(xc, yc_bernoulli), (xc, yc_precip)],
            B.cast(torch.float32, hmap_X),
            )
    hmap_bernoulli = B.to_numpy(hmap_dist.bernoulli_prob) # (1, 1, num^2)
    hmap_kappa, hmap_chi = hmap_dist.z_gamma[..., 0:1, :], hmap_dist.z_gamma[..., 1:2, :] # both (1, 1, num^2)

    hmap_rain = np.zeros(hmap_bernoulli.shape[-1])
    for i, (kappa_i, chi_i, bern_i) in enumerate(zip(hmap_kappa.reshape(-1), hmap_chi.reshape(-1), hmap_bernoulli.reshape(-1))):
        if bern_i > 0.5:
            sample_i = np.mean(np.random.gamma(shape=kappa_i.detach().cpu().numpy(), scale=chi_i.detach().cpu().numpy(), size=num_samples))
            hmap_rain[i] = sample_i
        else:
            hmap_rain[i] = 0
    hmap_rain = hmap_rain.reshape((num, num)) # (num^2,) => (num, num)

    # Plot the Heatmap with generated precipitation predictions over gridded inputs, with or without the true precipitation next to it for comparison
    if reference:
        _, (ax1, ax2) = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (10,20))
        plot1 = ax1.imshow(hmap_rain, cmap='plasma', alpha=0.5, vmin=0, vmax=np.max(B.to_numpy(batch['reference'][0])), extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]])
        ax1.scatter(B.to_numpy(batch['xc'])[0, 0], B.to_numpy(batch['xc'])[0, 1], marker = 'o', color='k', label='Context Points', s=0.1)
        ax1.set_title(f'Model Predicted Rainfall')
        plot2 = ax2.imshow(batch['reference'][0], alpha = 0.5, cmap='plasma', vmin=0, vmax=np.max(B.to_numpy(batch['reference'][0])), extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]])
        ax2.set_title(f'True Rainfall')
        plt.colorbar(plot1, ax=ax1, shrink=0.15)
        plt.colorbar(plot2, ax=ax2, shrink=0.15)
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()
    else:
        plt.imshow(hmap_rain, cmap='plasma', alpha=0.5, vmin=0)
        plt.scatter(B.to_numpy(batch['xc'])[0, 0], B.to_numpy(batch['xc'])[0, 1], marker = 'o', color='k', label='Context Points', s=0.1)
        plt.colorbar()
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()