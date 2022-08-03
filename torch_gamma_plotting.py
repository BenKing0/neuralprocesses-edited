import lab as B
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import numpy as np
from tqdm import tqdm


def plot_1d():
    return


def plot_2d(
    state: B.RandomState, 
    model, 
    generator, 
    save_path: str, 
    xbounds: Tuple[float, float], 
    ybounds: Tuple[float, float], 
    reference: bool = False,
    device: str = 'cpu',
    num_samples: int = 1,
    ):
    sns.set_theme()

    with B.on_device(device):
        epoch = generator.epoch()
        batch = epoch[0]    

    num = 200
    hmap_x1 = np.linspace(xbounds[0], xbounds[1], num)
    hmap_x2 = np.linspace(ybounds[0], ybounds[1], num)
    hmap_X = np.meshgrid(hmap_x1, hmap_x2)
    hmap_X = np.array(hmap_X).reshape((1, 2, num**2)) # (2, num, num) = (coords, xs, ys) to (1, 2, num^2)
    yc_precip = batch['yc'][0:1, 0:1, :] # (b, 1, n) to (1, 1, n)
    xc = batch['xc'][0:1] # (b, 2, n) to (1, 2, n)

    with torch.no_grad():
        _, hmap_dist = model(
            state, 
            xc, 
            yc_precip,
            B.cast(torch.float32, hmap_X),
            )
    hmap_kappa, hmap_chi = hmap_dist.kappa, hmap_dist.chi # both (1, 1, num^2)

    hmap_rain = np.zeros(hmap_kappa.shape[-1])
    for i, (kappa_i, chi_i) in enumerate(zip(hmap_kappa.reshape(-1), hmap_chi.reshape(-1))):
        sample_i = np.mean(np.random.gamma(shape=kappa_i.detach().cpu().numpy(), scale=1/chi_i.detach().cpu().numpy(), size=num_samples))
        hmap_rain[i] = sample_i
    hmap_rain = hmap_rain.reshape((num, num)) # (num^2,) => (num, num)

    _, ref_vals = list(zip(*batch['reference'][0]))
    ref_vals = np.array(ref_vals).reshape((30, 30)) # 30 hardcoded into reference

    if reference:
        _, (ax1, ax2) = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (10,20))
        plot1 = ax1.imshow(hmap_rain, cmap='plasma', alpha=0.5, vmin=0, vmax=np.max(ref_vals), extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]])
        ax1.scatter(B.to_numpy(batch['xc'])[0, 0], B.to_numpy(batch['xc'])[0, 1], marker = 'o', color='k', label='Context Points', s=0.2)
        ax1.set_title(f'Model Predicted')
        plot2 = ax2.imshow(ref_vals, alpha = 0.5, cmap='plasma', vmin=0, vmax=np.max(ref_vals), extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]])
        ax2.set_title(f'Ground Truth')
        plt.colorbar(plot1, ax=ax1, shrink=0.15)
        plt.colorbar(plot2, ax=ax2, shrink=0.15)
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()
    else:
        plt.imshow(hmap_rain, cmap='plasma', alpha=0.5, vmin=0)
        plt.scatter(B.to_numpy(batch['xc'])[0, 0], B.to_numpy(batch['xc'])[0, 1], marker = 'o', color='k', label='Context Points', s=0.2)
        plt.colorbar()
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()