import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lab as B
import torch
import numpy as np
import os
from tqdm import trange
from scipy.ndimage import gaussian_filter
sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})


def run(
    run_num, 
    model_bern_gamma_joint = ['ConvCBNP', 'ConvCGNP', 'ConvCNP'], 
    xbounds = [0, 60], 
    ybounds = [0, 60],
    ):

    loc_gamma = f'/real_rainfall/x2_y1/{model_bern_gamma_joint[1]}/unet/elbo/gamma-real-results-{run_num}.csv'
    loc_bernoulli = f'/real_rainfall/x2_y1/{model_bern_gamma_joint[0]}/unet/elbo/bernoulli-real-results-{run_num}.csv'
    save_dir = f'/real_rainfall/x2_y1/{model_bern_gamma_joint[2]}/unet/elbo/'
    save_file = save_dir + f'run-{run_num}.png'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gamma_df = pd.read_csv(loc_gamma)
    bernoulli_df = pd.read_csv(loc_bernoulli)

    # of shape (n(=200), n) and (m(=30), m) for n, m the number of 1D points, with channel 1 being the projection of gamma vals to bernoulli
    pred_ys, ref_ys = merge(bernoulli_df, gamma_df) 
    plot(
        pred_ys,
        ref_ys,
        save_file,
        xbounds,
        ybounds,
    )

    return


def merge(bernoulli_df, gamma_df):

    bern_pred = np.where(bernoulli_df['rain'].dropna() > 0.5, 1, 0)
    gamma_pred = np.array(gamma_df['rain'].dropna())
    assert bern_pred.shape == gamma_pred.shape

    # of shapes (n(=200)^2, ) and (m(=30)^2, )
    pred_ys = bern_pred * gamma_pred
    assert pred_ys.shape == gamma_pred.shape
    ref_ys = np.array(gamma_df['rain_ref'].dropna())

    n = int(np.sqrt(pred_ys.shape[0]))
    pred_ys = pred_ys.reshape(n, n)

    m = int(np.sqrt(ref_ys.shape[0]))
    ref_ys = ref_ys.reshape(m, m)

    return pred_ys, ref_ys


def plot(
    pred_ys, 
    ref_ys, 
    save_file,
    xbounds,
    ybounds,
    ):
    '''
    pred_xs: (2, n^2) shape
    pred_ys: (n, n) shape
    ref_xs: (2, m^2) shape
    ref_ys: (m, m) shape
    save_dir: name and location of where figure saved
    xbounds: the x bounds of the plots
    ybounds: the y bounds of the plots
    '''

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    smoothed = gaussian_filter(pred_ys, sigma=1)

    plot1 = ax1.imshow(smoothed, cmap='PuOr', alpha=0.5, vmin=0, vmax=np.max(smoothed), extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]])
    ax1.set_title(f'Model Predicted')
    plot2 = ax2.imshow(ref_ys, alpha = 0.5, cmap='PuOr', vmin=0, vmax=np.max(ref_ys), extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]])
    ax2.set_title(f'Ground Truth')
    plt.colorbar(plot1, ax=ax1, shrink=1.)
    plt.colorbar(plot2, ax=ax2, shrink=1.)    
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    for i in trange(1, 366, 1):
        model_bern_gamma_joint = ['ConvCBNP', 'ConvCGNP', 'ConvCNP']
        try:
            run(i, model_bern_gamma_joint=model_bern_gamma_joint)
        except FileNotFoundError:
            pass