from scipy.stats import norm
import torch
import lab as B
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy.ndimage import gaussian_filter


def _group_classes(inputs, outputs, grouper):
    '''
    Like filter() with a list, but more condensed and faster.
    '''

    if inputs.ndim == 1:
        pred_df = pd.DataFrame({
            'x': inputs,
            'y': outputs, 
            'z': grouper,
        })
        groups = pred_df.groupby('z')
        return groups

    elif inputs.ndim > 1:
        input_indexer = B.linspace(0, inputs.shape[0]-1, inputs.shape[0]).astype(np.int32)
        groups = _group_classes(input_indexer, outputs, grouper)
        return {class_: inputs[group.x] for class_, group in groups}, groups

    else:
        print('Inputs have to be at least 1 dimensional.')
        return

    
##NOTE: only defined for binary classification
def plot_classifier_1d(state, model, gen, save_path, means, vars, prior: list = [0.5, 0.5], device='cpu'):

    with B.on_device(device):

        def true_boundary(x, means, vars):
            norm1 = norm(loc=means[1], scale=vars[1])
            numer = prior[1] * norm1.pdf(x)
            norm0 = norm(loc=means[0], scale=vars[0])
            Z = prior[0] * norm0.pdf(x) + numer
            return numer / Z

        epoch = gen.epoch()
        batch = epoch[0] ## arbitrarily take the first batch (task) from the epoch to plot

        # Predict with model.
        with torch.no_grad():
            state, dist = model(state, batch['xc'], batch['yc'], batch['xt']) ## 'dist' here is the BernoulliDistribution class defined above (only implemented for binary classification)
            class_1_probs = dist.probs
        
        rhos = B.squeeze(class_1_probs) ## (1, n) => (n, )
        membership = B.cast(torch.float32, rhos > 0.5)
        groups = _group_classes(B.to_numpy(B.squeeze(batch['xt'])), B.to_numpy(rhos), B.to_numpy(membership))

        ## generate smooth prediction line for visualisation purposes
        smooth_xs = B.expand_dims(B.linspace(B.min(*batch['xt']), B.max(*batch['xt']), 1000), axis=0)
        _, smooth_dist = model(state, batch['xc'], batch['yc'], B.cast(torch.float32, smooth_xs))
        _smooth_preds = smooth_dist.probs
        smooth_preds = B.squeeze(_smooth_preds)

        sns.set_theme()
        plt.scatter(batch['xt'], batch['yt'], marker='.', c='k', label='Targets')
        cs = ['xkcd:light red', 'xkcd:light blue']
        for class_, group in groups:
            plt.plot(group.x, group.y, '+', color=cs[class_], label=f'{class_} predicted')
        plt.plot(B.to_numpy(B.squeeze(smooth_xs)), B.to_numpy(smooth_preds), '-', color='xkcd:green', label='Smoothed')
        plt.plot(B.to_numpy(B.squeeze(smooth_xs)), true_boundary(B.cast(np.float32, B.squeeze(smooth_xs)), means, vars), 'k-', label='Truth')
        plt.ylim(-0.1, 1.1)
        plt.ylabel('Posterior Prob. of Class 1')
        plt.xlabel('x')
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


##NOTE: can handle multinomial classification when 'yc', 'yt' from 'gen' have >2 classes
def plot_classifier_2d(state, model, gen, save_path, means, vars, priors, hmap_class: int = 0, device='cpu'):
    '''
    Plot 2D xs belonging to 1 of K classes. Therefore dim_x = 2, dim_y = K.

    Args:
    --------------
    ...
    cmap_class: (int, defaults to 1) the class focused on when plotting the heat map. I.e. the colour of the heatmap refers to P(y=cmap_class | x).
    '''

    with B.on_device(device):

        epoch = gen.epoch()
        batch = epoch[0]    

        with torch.no_grad():
            state, dist = model(state, batch['xc'], batch['yc'], batch['xt']) 
            prob_vectors = dist.probs ## (k, n) for k classes and n target points

        np_batch = B.to_numpy(batch)
        np_probs = B.to_numpy(prob_vectors)
        if np_batch['xc'].shape[0] == 1:
            memberships = B.cast(torch.int32, prob_vectors > 0.5)
            true_memberships = B.cast(torch.int32, batch['yt'])
        else:
            memberships = B.argmax(prob_vectors, axis=0) ## convert (k, n) array of probabilities to (n, ) integers
            true_memberships = B.argmax(batch['yt'], axis=0)
        x_dict, groups = _group_classes(np.transpose(np_batch['xt']), np_probs[hmap_class], B.to_numpy(B.squeeze(true_memberships)))

        _num = 150
        _hmap_x = np.linspace(min(np_batch['xt'].flatten()), max(np_batch['xt'].flatten()), _num)
        _hmap_X = np.meshgrid(_hmap_x, _hmap_x) ## of shape (2, _num, _num) = (coords, xs, ys)
        hmap_X = np.array(_hmap_X).reshape((2, _num**2)) ## of shape (2, _num ^ 2) to be of (c, n) shape
        _, hmap_dist = model(state, batch['xc'], batch['yc'], B.cast(torch.float32, hmap_X))
        _hmap_probs = hmap_dist.probs[hmap_class] ## (k, _num ^ 2) for k classes and _num points: (k, _num ^ 2) => (_num ^ 2, ) when selecting hmap_class
        hmap_probs = B.to_numpy(_hmap_probs).reshape((_num, _num))

        sns.set_theme()
        sns.set_theme()
        plt.pcolormesh(_hmap_X[0], _hmap_X[1], hmap_probs, vmin=-0.1, vmax=1.1, cmap='RdYlBu', alpha=0.4)   
        plt.colorbar()
        markers = itertools.cycle(('x', '+', '.', 'o', '*'))
        for class_, _ in groups:
            x = x_dict[class_]
            plt.scatter(x[:,0], x[:,1], marker=next(markers), c='k', label=f'{class_} True')
        plt.legend()
        if B.max(memberships) == 1:
            plt.title('Colormap scaled by class 1')
        else:
            plt.title(f'Colormap scaled by class {hmap_class}')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    ##TODO: 
    # 1. add colors in plots to signify ground truth
    # 2. fix hmap
