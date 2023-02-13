import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from warnings import warn
import math
import torch

from data.dataset import Dataset

class myData(Dataset):
    
    def __init__(self, 
                 n_train: int = 400,
                 f=lambda x: x.mul(4).add(0.8).cos(),
                 fracs=[ 0.25, 0.25, 0.25], 
                 stds=[ 0.0, 0.0, 0.5], 
                 spreads=[ 4.5,  4.5, 4.5], 
                 offsets=[ -6.5, 2.5, -2], 
                 train_inter=[-6.5, 6.5], 
                 test_inter=[-6.5, 6.5]):
        
        super().__init__()
        
        shuffle = True
        mini_batch_size = 1
        n_mini_batches = math.ceil(n_train/mini_batch_size) # batches per epoch

        x_train, y_train, x_test, y_test = self.make_data(
            n_train,
            shuffle_train=shuffle,
            f=f, # x.pow(3)/350, 
            fracs =  fracs,
            stds = stds,
            spreads = spreads,
            offsets = offsets, 
            right_lim = train_inter[-1]
        )
        
        in_data = np.vstack([x_train, x_test])
        out_data = np.vstack([y_train, y_test])
        
        # Specify internal data structure.
        self._data['classification'] = False
        self._data['sequence'] = False
        self._data['in_data'] = in_data
        self._data['in_shape'] = [1]
        self._data['out_data'] = out_data
        self._data['out_shape'] = [1]
        self._data['train_inds'] = np.arange(len(y_train))
        self._data['test_inds'] = np.arange(len(y_train), len(y_train) + len(y_test))
        
        self._map = lambda x:x.cos()
        print(train_inter, test_inter)
        self._train_inter = train_inter
        self._test_inter = test_inter
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def make_data(self, n_train: int = 400,
        f=lambda x: x.mul(4).add(0.8).cos(),
        fracs=[1.0], # , 0.7], 
        stds = [0.0], # , 0.01],
        spreads = [6],
        offsets = [-3],
        shuffle_train=True,
        right_lim = 6.5,
    ):
        """
        Create some clusters of data points.
        Arguments:
            fracs - fraction of n_train per cluster
            stds - cluster y stds
            spreads - cluster x stds
            offsets - cluster offsets
        """
        n_clusters = len(fracs)
        assert len(fracs) == len(stds) == len(spreads) == len(offsets)
        left_lim = -right_lim
        
        offset_x = 0
        # assert n_train % n_clusters == 0
        
        # make some clusters of data points
        cluster_shapes = [(int(n_train * fracs[i]),  1) for i in range(n_clusters)] # evenly distributes datapoints among clusters

        clusters = []
        for i in range(n_clusters):
            xi = torch.rand(*cluster_shapes[i]) * spreads[i] + offset_x + offsets[i]
            clusters += [xi]
            
        x_train = torch.cat(clusters)
        
        # due to fracs, x_train may be shorter than n_train now
        n_train = len(x_train)
        
        # give clusters different variances to investigate if heteroskedastic gaussian approximates them better
        noise = torch.cat([stds[i] * torch.randn(*cluster_shapes[i]) for i in range(n_clusters)])
        
        # apply our function and add ("aleatoric") noise
        y_train = f(x_train) + noise
        
        if shuffle_train:
            rand_order = torch.randperm(n_train)
            x_train = x_train[rand_order]
            y_train = y_train[rand_order]
        
        # test/visualise on whole x axis
        x_test = torch.linspace(left_lim+offset_x, right_lim+offset_x, 1000).unsqueeze(-1)
        y_test = f(x_test) # no noise
        y_train = y_train
        
        return x_train, y_train, x_test, y_test
    
    def plot_dataset(self, show=True):
        """Plot the whole dataset.

        Args:
            show: Whether the plot should be shown.
        """

        train_x = self.get_train_inputs().squeeze()
        train_y = self.get_train_outputs().squeeze()

        test_x = self.get_test_inputs().squeeze()
        test_y = self.get_test_outputs().squeeze()

        if self.num_val_samples > 0:
            val_x = self.get_val_inputs().squeeze()
            val_y = self.get_val_outputs().squeeze()

        sample_x, sample_y = self._get_function_vals()

        # The default matplotlib setting is usually too high for most plots.
        plt.locator_params(axis='y', nbins=2)
        plt.locator_params(axis='x', nbins=6)

        plt.plot(sample_x, sample_y, color='k', label='f(x)',
                 linestyle='dashed', linewidth=.5)
        plt.scatter(train_x, train_y, color='r', label='Train')
        plt.scatter(test_x, test_y, color='b', label='Test', alpha=0.8)
        if self.num_val_samples > 0:
            plt.scatter(val_x, val_y, color='g', label='Val', alpha=0.5)
        plt.legend()
        plt.title('1D-Regression Dataset')
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        if show:
            plt.show()
            
    @property
    def train_x_range(self):
        """Getter for read-only attribute train_x_range."""
        return self._train_inter

    @property
    def test_x_range(self):
        """Getter for read-only attribute test_x_range."""
        return self._test_inter

    @property
    def val_x_range(self):
        """Getter for read-only attribute val_x_range."""
        return self._val_inter
    
    def _get_function_vals(self, num_samples=100, x_range=None):
        """Get real function values for x values in a range that
        covers the test and training data. These values can be used to plot the
        ground truth function.

        Args:
            num_samples: Number of samples to be produced.
            x_range: If a specific range should be used to gather function
                values.

        Returns:
            x, y: Two numpy arrays containing the corresponding x and y values.
        """

        return self.x_test, self.y_test
            

    def plot_dataset(self, show=True):
        """Plot the whole dataset.

        Args:
            show: Whether the plot should be shown.
        """

        train_x = self.get_train_inputs().squeeze()
        train_y = self.get_train_outputs().squeeze()

        test_x = self.get_test_inputs().squeeze()
        test_y = self.get_test_outputs().squeeze()

        if self.num_val_samples > 0:
            val_x = self.get_val_inputs().squeeze()
            val_y = self.get_val_outputs().squeeze()

        sample_x, sample_y = self._get_function_vals()

        # The default matplotlib setting is usually too high for most plots.
        plt.locator_params(axis='y', nbins=2)
        plt.locator_params(axis='x', nbins=6)

        plt.plot(sample_x, sample_y, color='k', label='f(x)',
                 linestyle='dashed', linewidth=.5)
        plt.scatter(train_x, train_y, color='r', label='Train')
        plt.plot(test_x, test_y, color='b', label='Test', alpha=0.8)
        if self.num_val_samples > 0:
            plt.scatter(val_x, val_y, color='g', label='Val', alpha=0.5)
        plt.legend()
        plt.title('1D-Regression Dataset')
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        if show:
            plt.show()


    def get_identifier(self):
        """Returns the name of the dataset."""
        return '1DRegression'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):

        raise NotImplementedError('TODO implement')

if __name__ == '__main__':
    pass


