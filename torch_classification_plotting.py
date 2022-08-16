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

    
# NOTE: only defined for binary classification
def plot_classifier_1d(state, model, gen, save_path, means=None, vars=None, prior: list = [0.5, 0.5], device='cpu', num_samples=25):

    with B.on_device(device):

        def true_boundary(x, means, vars):
            norm1 = norm(loc=means[1], scale=vars[1])
            numer = prior[1] * norm1.pdf(x)
            norm0 = norm(loc=means[0], scale=vars[0])
            Z = prior[0] * norm0.pdf(x) + numer
            return numer / Z

        epoch = gen.epoch()
        batch = epoch[0] ## arbitrarily take the first epoch to plot

        # Predict with model.
        with torch.no_grad():
            state, dist = model(state, batch['xc'], batch['yc'], batch['xt']) ## 'dist' here is the BernoulliDistribution class defined above (only implemented for binary classification)
            class_1_probs = dist.probs[0] # (1, 1, n) to (1, n)
        
        rhos = B.squeeze(class_1_probs) # (1, n) => (n, )
        membership = B.cast(torch.float32, rhos > 0.5)
        groups = _group_classes(B.to_numpy(B.squeeze(batch['xt'][0])), B.to_numpy(rhos), B.to_numpy(membership))

        ## generate smooth prediction line for visualisation purposes
        smooth_xs = B.expand_dims(B.linspace(B.min(*batch['xt'][0].cpu()), B.max(*batch['xt'][0].cpu()), 1000), axis=0)
        smooth_preds_multi = []
        for _ in range(num_samples):
            _, smooth_dist = model(state, batch['xc'], batch['yc'], B.cast(torch.float32, smooth_xs))
            _smooth_preds = smooth_dist.probs[0]
            smooth_preds_i = B.squeeze(_smooth_preds)
            smooth_preds_multi.append(smooth_preds_i.cpu().detach().numpy())
        smooth_preds = np.mean(smooth_preds_multi, axis=0)
        smooth_preds_stddev = np.std(smooth_preds_multi, axis=0)

        sns.set_theme()
        plt.scatter(batch['xc'][0].T, batch['yc'][0].T, marker='.', c='k', label='Context set')
        plt.plot(B.to_numpy(B.squeeze(smooth_xs)), B.to_numpy(smooth_preds), '-', color='xkcd:green', label='Smoothed')
        plt.fill_between(
            B.to_numpy(B.squeeze(smooth_xs)), 
            B.to_numpy(smooth_preds)+1.96*B.to_numpy(smooth_preds_stddev), 
            B.to_numpy(smooth_preds)-1.96*B.to_numpy(smooth_preds_stddev), 
            alpha=0.4,
            color='xkcd:green')
        if means and vars:
            plt.plot(B.to_numpy(B.squeeze(smooth_xs)), true_boundary(B.cast(np.float32, B.squeeze(smooth_xs)), means, vars), 'k-', label='Truth')
        plt.ylim(-0.1, 1.1)
        plt.ylabel('Posterior Prob. of Class 1')
        plt.xlabel('x')
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


# NOTE: can handle multinomial classification when 'yc', 'yt' from 'gen' have > 2 classes
def plot_classifier_2d(state, model, gen, save_path, hmap_class: int = 0, device='cpu', num_samples=25, data_name=None, xbounds=[0, 60], ybounds=[0, 60]):
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
            prob_vectors = dist.probs[0, 0] ## (k, n) for k classes and n target points

        np_batch = {key: B.to_numpy(value) for key, value in batch.items() if key != 'reference'}
        np_probs = B.to_numpy(prob_vectors)
        if batch['yt'][0].shape[1] == 1:
            memberships = B.cast(torch.int32, prob_vectors > 0.5)
            true_memberships = B.cast(torch.int32, batch['yt'][0])
        else:
            memberships = B.argmax(prob_vectors, axis=0) ## convert (k, n) array of probabilities to (n, ) integers
            true_memberships = B.argmax(batch['yt'][0], axis=0)
        x_dict, groups = _group_classes(np.transpose(np_batch['xt'][0]), np_probs[hmap_class], B.to_numpy(B.squeeze(true_memberships)))

        num = 200
        hmap_x = np.linspace(xbounds[0], xbounds[1], num)
        hmap_y = np.linspace(ybounds[0], ybounds[1], num)
        hmap_X = np.meshgrid(hmap_x, hmap_y) ## of shape (2, _num, _num) = (coords, xs, ys)
        hmap_X = np.array(hmap_X).reshape((1, 2, num**2)) ## of shape (1, 2, _num ^ 2) to be of (b, c, n) shape
        hmap_probs_multi = []
        for _ in range(num_samples):
            _, hmap_dist = model(state, batch['xc'], batch['yc'], B.cast(torch.float32, hmap_X))
            hmap_probs_i = hmap_dist.probs[0][hmap_class] ## (k, _num ^ 2) for k classes and _num points: (k, _num ^ 2) => (_num ^ 2, ) when selecting hmap_class
            hmap_probs_multi.append(hmap_probs_i.cpu().detach().numpy())
        hmap_probs = np.mean(hmap_probs_multi, axis=0)
        hmap_probs = B.to_numpy(hmap_probs).reshape((num, num))

        try:
            ref_xs, ref_ys = list(zip(*batch['reference'][0])) # this is a list(zip(ref_xs, ref_ys))
            ref_ys = np.array(ref_ys).reshape((30, 30)) # number of reference points hard-coded to 30
            ref_probs = []
            for _ in range(num_samples):
                _, ref_dist = model(state, batch['xc'], batch['yc'], B.cast(torch.float32, np.array(ref_xs)).reshape(2, -1))
                ref_probs.append(ref_dist.probs[0][hmap_class])

            ref_probs = torch.mean(torch.stack(ref_probs), axis=0)
            memberships = torch.where(ref_probs > 0.5, 1., 0.)
            correct = torch.where(memberships==torch.tensor(ref_ys).to(device).flatten(), 1, 0)
            accuracy = torch.sum(correct.flatten()) / len(correct.flatten())
        
        except ValueError:
            ref_ys = batch['reference'][0]
            ref_ys = np.array(ref_ys).reshape((30, 30)) # number of reference points hard-coded to 30
            ref_ys = np.where(ref_ys > 0, 1, 0)

        sns.set_theme()
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
        plot1 = ax1.imshow(hmap_probs, vmin=-0.1, vmax=1.1, cmap='RdYlBu', extent=[min(hmap_X[0, 0]), max(hmap_X[0, 0]), min(hmap_X[0, 1]), max(hmap_X[0, 1])], alpha=0.4)  
        plot2 = ax2.imshow(ref_ys, vmin=-0.1, vmax=1.1, cmap='RdYlBu', extent=[min(hmap_X[0, 0]), max(hmap_X[0, 0]), min(hmap_X[0, 1]), max(hmap_X[0, 1])], alpha=0.4)   
        plt.colorbar(plot1, ax=ax1)
        plt.colorbar(plot2, ax=ax2)
        markers = itertools.cycle(('.', '.', 'o', '+', '*'))
        for class_, _ in groups:
            x = x_dict[class_]
            ax1.scatter(x[:,0], x[:,1], marker=next(markers), c='k', s=0.5)
        # ax1.text(5, 5, f'Accuracy: {accuracy:.2f}')
        if B.max(memberships) == 1:
            ax1.set_title('Colormap scaled by class 1')
        else:
            ax1.set_title(f'Colormap scaled by class {hmap_class}')
        ax2.set_title('Reference')
        plt.savefig('/'.join(save_path.split('/'))+f'', bbox_inches='tight', dpi=300)
        plt.close()

        if data_name == 'synthetic':
            directory = '/'.join(save_path.split('/')[:-1])
            file_id = gen.i
            filename = directory + f'/bernoulli-results-{file_id}.csv'

            ref_xs = np.linspace(0, 58, 30)
            ref_xs = np.meshgrid(ref_xs, ref_xs) ## of shape (2, 30, 30) = (coords, xs, ys)
            ref_xs = np.array(ref_xs).reshape((2, 30**2)) ## of shape (2, 30 ^ 2)

            contents = {
                'x': pd.Series(hmap_X[0][0]),
                'y': pd.Series(hmap_X[0][1]),
                'rain': pd.Series(hmap_probs.reshape(num**2)),
                'xref': pd.Series(ref_xs[0]),
                'yref': pd.Series(ref_xs[1]),
                'rain_ref': pd.Series(ref_ys.reshape(900)),
            }

            file = pd.DataFrame(contents)
            file.to_csv(filename)

        elif data_name == 'real_rainfall':
            directory = '/'.join(save_path.split('/')[:-1])
            file_id = gen.i
            filename = directory + f'/bernoulli-real-results-{file_id}.csv'

            ref_xs = np.linspace(0, 58, 30)
            ref_xs = np.meshgrid(ref_xs, ref_xs) ## of shape (2, 30, 30) = (coords, xs, ys)
            ref_xs = np.array(ref_xs).reshape((2, 30**2)) ## of shape (2, 30 ^ 2)

            contents = {
                'x': pd.Series(hmap_X[0][0]),
                'y': pd.Series(hmap_X[0][1]),
                'rain': pd.Series(hmap_probs.reshape(num**2)),
                'xref': pd.Series(ref_xs[0]),
                'yref': pd.Series(ref_xs[1]),
                'rain_ref': pd.Series(ref_ys.reshape(900)),
            }

            file = pd.DataFrame(contents)
            file.to_csv(filename)