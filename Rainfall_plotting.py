import lab as B
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from lab import Dispatcher


_dispatch = Dispatcher()


@_dispatch
def rainfall_plotter(
    state, 
    model, 
    generator, 
    save_path: str, 
    xbounds: Tuple[float, float], 
    ybounds: Tuple[float, float], 
    reference: bool = False,
    device: str ='cpu',
    ):
    sns.set_theme()

    with B.on_device(device):
        epoch = generator.epoch()
        batch = epoch[0]    

    np_batch = batch.to_numpy()    

    with torch.no_grad():
        state, dist = model(state, batch['xc'], batch['yc'], batch['xt']) 
        rain_amount = dist.output[0]

    # Build grid for plotting model rainfall predictions spatially (not just context and target outputs and predictions)
    _num = 200
    _hmap_x1 = np.linspace(xbounds[0], xbounds[1], _num)
    _hmap_x2 = np.linspace(ybounds[0], ybounds[1], _num)
    _hmap_X = np.meshgrid(_hmap_x1, _hmap_x2) # of shape (2, _num, _num) = (coords, xs, ys)
    hmap_X = np.array(_hmap_X).reshape((2, _num**2)) # of shape (2, _num ^ 2) to be of (c, n) shape
    _, _hmap_output = model(state, batch['xc'], batch['yc'], B.cast(torch.float32, hmap_X))
    hmap_outputs = _hmap_output.output[0]
    hmap_outputs = B.to_numpy(hmap_outputs).reshape((_num, _num)) # (1, _num ^ 2) for 1 output dimension and _num points: (1, _num ^ 2) => (_num ^ 2, ) when selecting hmap_class

    # Plot the Heatmap with generated precipitation predictions over gridded inputs, with or without the true precipitation next to it for comparison
    if reference:
        _, (ax1, ax2) = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (5,10))
        ax1.pcolormesh(_hmap_X[0], _hmap_X[1], hmap_outputs, vmin = 0, cmap = 'Pastel2', alpha = 0.4, shading = 'auto')   
        ax1.scatter(np_batch['xc'][:, 0], np_batch['xc'][:, 1], marker = 'o', color='k', label='Context Points')
        ax1.scatter(np_batch['xt'][:, 0], np_batch['xt'][:, 1], marker = '+', color='k', label='Target Points')
        ax1.colorbar()
        ax1.set_title(f'Model Predicted Rainfall')
        ax2.imshow(batch['reference'], alpha = 0.4)
        ax2.set_cmap('Pastel2')
        ax2.colorbar()
        ax2.set_title(f'True Rainfall')
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()
    else:
        plt.pcolormesh(_hmap_X[0], _hmap_X[1], hmap_outputs, vmin = 0, cmap = 'Pastel2', alpha = 0.4, shading = 'auto')   
        plt.colorbar()
        plt.scatter(np_batch['xc'][:, 0], np_batch['xc'][:, 1], marker = 'o', color='k', label='Context Points')
        plt.scatter(np_batch['xt'][:, 0], np_batch['xt'][:, 1], marker = '+', color='k', label='Target Points')
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
        plt.close()
        

@_dispatch
def rainfall_plotter(
    *, 
    model, 
    generator, 
    save_path: str, 
    xbounds: Tuple[float, float], 
    ybounds: Tuple[float, float], 
    reference: np.ndarray = None, 
    device: str = 'cpu'
    ):
    state = B.create_random_state(torch.float32, seed = 0)
    return rainfall_plotter(state, model, generator, save_path, xbounds, ybounds, reference, device)