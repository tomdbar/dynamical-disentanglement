import itertools
import os
from abc import ABC, abstractmethod

import PIL
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from matplotlib import gridspec


class FactorSpace():
    '''Stores and indexes the allowed configurations of generative factors for a dataset.'''

    def __init__(self, factor_vals, factor_names=None):
        '''Stores and indexes the allowed configurations of generative factors for a dataset.

        Args:
            factor_vals: A list of values allowed for each factor.  Each set of values can either
                         be passed as a list, [v_1, v_2, ...], or a single integer, n, such that
                         the allowed values are [0,1,...n-1] for n>1.
            factor_names: A list of names for the configured factors.  Default names are generated
                          if none are passed.
        '''
        def _to_list(x):
            if type(x) is not list:
                try:
                    x = list(x)
                except TypeError:
                    x = list(range(x))
            return x

        self.factor_vals = [_to_list(val) for val in factor_vals]
        self._factor_sizes = np.array([len(v) for v in self.factor_vals])
        self._max_factor_vals = np.array([max(v) for v in self.factor_vals])
        self._min_factor_vals = np.array([min(v) for v in self.factor_vals])
        self.num_dims = len(self.factor_vals)

        if factor_names is not None:
            assert len(factor_names) == self.num_dims, "Length of factor names doesn't match factor vals."
            self.factor_names = factor_names
        else:
            self.factor_names = [f"factor_{i}" for i in range(self.num_dims)]

        self.__basis = None

    def __len__(self):
        '''The number of possible unique configurations of factors.'''
        return int(np.prod(self._factor_sizes))

    def get_all_factors(self):
        '''Get all possible configurations of factors.

        Returns: An array of all possible configurations of factors.
        '''
        if self.__basis is None:
            self.__basis = np.array(list(itertools.product(*self.factor_vals)))
        return self.__basis

    def _validate_factors(self, factors):
        '''Check that a set of factors exists within the factor space.

        If the factors do not exist within this FactorSpace, a value error
        is raised.

        Args:
            factors: A factor configuration (or array of configurations).

        Returns: The passed factors, if valid.
        '''
        if type(factors) is not np.ndarray:
            factors = np.array(factors)
        if factors.ndim == 1:
            factors = np.expand_dims(factors, 0)
        assert factors.shape[-1] == self.num_dims, "Factors do not have the valid dimension."

        if (any(np.min(factors, axis=0) < self._min_factor_vals)
                or any(np.max(factors, axis=0) > self._max_factor_vals)):
            raise ValueError("Invalid factors passed.")

        return factors

    def factor2idx(self, factors):
        '''Convert factors to their index in the factor space.

        We index, idx, all unique factor configurations, such that self.__basis[idx] = factors.

        Args:
            factors: A factor configuration (or array of configurations).

        Returns: An array of indices.
        '''
        factors = self._validate_factors(factors)
        idxs = np.zeros(len(factors))
        for idx_dim in range(factors.shape[-1] - 1, -1, -1):
            factor_idxs = np.searchsorted(self.factor_vals[idx_dim], factors[..., idx_dim])
            idxs += max(np.prod(self._factor_sizes[idx_dim + 1:]), 1) * factor_idxs

        return idxs.astype(int)

class FactorisedDataset(ABC):
    '''A dataset with identifiable underlying generative factors taking on a finite set of discrete values.'''

    def __init__(self,
                 factor_vals,
                 factor_names=None,
                 use_torch=False):
        '''Basic construction common to all FactorisedDataset's.

        Args:
            factor_vals: A list of values allowed for each factor.  Each set of values can either
                         be passed as a list, [v_1, v_2, ...], or a single integer, n, such that
                         the allowed values are [0,1,...n-1] for n>1.
            factor_names: A list of names for the configured factors.  Default names are generated
                          if none are passed.
            use_torch: Whether to use numpy or torch for arrays/tensors.
        '''
        if factor_names is None:
            factor_names = self._all_factor_names
        self.factor_space = FactorSpace(factor_vals, factor_names)
        self.dataset = self._load_dataset()
        if use_torch:
            self.to_torch()

        print("Intitialised dataset with following (factors : values)...")
        for factor, vals in zip(self.factor_space.factor_names, self.factor_space.factor_vals):
            print(f"\t{factor} : {vals}")

    @property
    @abstractmethod
    def _all_factor_names(self):
        '''A list of names for the underlying generative factors.'''
        raise NotImplemented

    @property
    @abstractmethod
    def _all_factor_dims(self):
        '''A list of the number of allowed values for each generative factor.'''
        raise NotImplemented

    @abstractmethod
    def _load_dataset(self):
        '''Load the dataset as an array of observation tensors.

        The index of an observation corresponds to the index of the underlying
        factor configuration in the FactorSpace.
        '''
        raise NotImplemented

    @classmethod
    def info(cls, as_str=False):
        '''Get a basic summary of the dataset.

        Args:
            as_str: If true, return the summary as a string, else it will be printed
                    and nothing returned.

        Returns: The summary string is as_str is True, else None.
        '''
        info_str = [f"{cls.__name__} is FactorisedDataset with 'factor name : number of allowed values':"]
        for name, dim in zip(cls._all_factor_names, cls._all_factor_dims):
            info_str.append(f"\t{name} : {dim}")
        info_str = "\n".join(info_str)
        if as_str:
            return info_str
        else:
            print(info_str)

    def get_data(self, factors):
        '''Get the data corresponding to the specified factor configurations.

        Args:
            factors: A factor configuration (or array of configurations).

        Returns: An array/tensor of the data values.
        '''
        idxs = self.factor_space.factor2idx(factors)
        return self.dataset[idxs]

    def factors2string(self, factors):
        '''Format the passed factors as a verbose string, identifying to
        which property they correspond.'''
        return "\n".join([f"{name} : {val}" for name, val in zip(self.factor_space.factor_names, factors)])

    def to_torch(self, device="cpu"):
        '''Convert the cached dataset to a torch tensor.

        Args:
            device: The device to map the tensor to.  Should be "cpu" or "cuda".
        '''
        if not torch.is_tensor(self.dataset):
            self.dataset = torch.from_numpy(self.dataset)
        self.dataset = self.dataset.to(device)

    def to_numpy(self):
        '''Convert the cached dataset to a numpy array.'''
        if not type(self.dataset) is np.ndarray:
            self.dataset = np.array(self.dataset)

    def imshow(self, factors=None):
        '''Visualise the data for a set of factors.

        If no factors are specified, all possible configurations will be plotted
        (not a great idea for big datasets!).

        Args:
            factors: A factor configuration (or array of configurations).

        Returns: A matplotlib.figure
        '''
        if factors is None:
            factors = self.factor_space.get_all_factors()
        try:
            len(factors[0])
        except:
            factors = [factors]
        num_axs = len(factors)

        n_row = num_axs // 5 + 1
        n_col = min(5, num_axs)

        fig = plt.figure(figsize=(3 * n_col, 3 * n_row))
        spec = gridspec.GridSpec(ncols=n_col, nrows=n_row)

        for f, s in zip(factors, spec):
            ax = fig.add_subplot(s)

            ax.imshow(self.get_data(f).squeeze())

            ax.set_title(self.factors2string(f), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for pos in ['bottom', 'top', 'right', 'left']:
                ax.spines[pos].set_color('0')

        fig.tight_layout()

        return fig

class GridWorld(FactorisedDataset):
    '''An n-dimensional grid with cyclical boundaries.

    The images are 64x64 with 3 (RGB) channels.
    '''

    _all_factor_names = ['North-South', 'East-West', 'Size', 'Color', 'Background Color']
    _all_factor_dims = [1000, 1000, 10, 5, 5]

    def __add_square(self, arr, center, size, col):
        '''Helper function for adding a square point to the grid.

        Args:
            arr: The array to add the point to.
            center: The position to add the point.
            size: The size of the point.
            col: The colour of the point.

        Returns: The input array with point added.
        '''
        l, r = center[0] - size/2, center[0] + size/2
        b, t = center[1] - size/2, center[1] + size/2
        arr[int(l):int(r),int(b):int(t)] = col
        return arr

    def __add_circle(self, arr, center, size, col):
        '''Helper function for adding a circular point to the grid.

        Args:
            arr: The array to add the point to.
            center: The position to add the point.
            size: The size of the point.
            col: The colour of the point.

        Returns: The input array with point added.
        '''
        Y, X = np.ogrid[:arr.shape[0], :arr.shape[0]]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = (dist_from_center < size/2)
        arr[mask] = col
        return arr

    def _load_dataset(self):
        '''Load the dataset as an array of observation tensors.

        The index of an observation corresponds to the index of the underlying
        factor configuration in the FactorSpace.
        '''
        dataset = np.zeros((len(self.factor_space),64,64,3))

        i_max, j_max = self.factor_space._factor_sizes[:2]

        diameter = (i_max + j_max) / 2

        mesh_step = int(64/(diameter))
        mesh_offset  = 0.5 * (64 - (mesh_step * diameter))

        def idx2color(idx):
            idx = int(idx)
            if idx == 0: # white
                return np.array([1,1,1])
            elif idx == 1: # red
                return np.array([1,0,0])
            elif idx == 2: # green
                return np.array([0,1,0])
            elif idx == 3: # blue
                return np.array([0,0,1])
            else: # idx == 4 --> black
                return np.array([0,0,0])

        idx = 0
        for i in range(i_max):
            for j in range(j_max):
                for size in self.factor_space.factor_vals[2]:
                    for col_idx in self.factor_space.factor_vals[3]:
                        for bkg_idx in self.factor_space.factor_vals[4]:
                            c = [(i + 0.5) * mesh_step + mesh_offset, (j + 0.5) * mesh_step + mesh_offset]
                            d = mesh_step * (1 - size/(self._all_factor_dims[2]+1))

                            dataset[idx][...] = idx2color(bkg_idx)
                            dataset[idx] = self.__add_square(dataset[idx], c, d, idx2color(col_idx))

                            idx += 1

        return dataset

class Cars3D(FactorisedDataset):
    '''Cars3D dataset.

    Images of cars with the variable viewing angle ('elevation', 'azimuth' factors) and
    car type ('object' factor) varying.  The images are formatted as  64x64 with 3 (RGB)
    channels.

    The data set was first used in the paper "Deep Visual Analogy-Making"
    (https://papers.nips.cc/paper/5845-deep-visual-analogy-making).  It is publicly
    available at : http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
    '''

    _path = "./_data/3dcars"
    _all_factor_names = ['elevation', 'azimuth', 'object']
    _all_factor_dims = [4, 23, 199]

    def _load_mesh(self, filename):
        '''Parses a single source file and rescales contained images

        Credit: this helper function is taken from Google Research's
        disentanglement-lib(data/ground_truth/cars3d.py).
        '''
        mesh = np.einsum("abcde->deabc", sio.loadmat(filename)["im"])
        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
        for i in range(flattened_mesh.shape[0]):
            pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
            pic.thumbnail((64, 64, 3), PIL.Image.ANTIALIAS)
            rescaled_mesh[i, :, :, :] = np.array(pic)
        return rescaled_mesh * 1. / 255

    def _load_dataset(self):
        '''Load the dataset as an array of observation tensors.

        The index of an observation corresponds to the index of the underlying
        factor configuration in the FactorSpace.
        '''
        dataset = np.zeros((len(self.factor_space), 64, 64, 3))

        object_idx_strings = [f"_{obj_idx+1:03d}_" for obj_idx in self.factor_space.factor_vals[2]]  # object
        object_files = [os.path.join(self._path, f) for f in os.listdir(self._path)
                        if (any(idx_str in f for idx_str in object_idx_strings)
                            and (os.path.splitext(f)[-1] == ".mat"))
                        ]

        for val, filename in zip(self.factor_space.factor_vals[2], object_files):
            data_mesh = self._load_mesh(filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(val,
                        len(factor1) * len(factor2))
            ])

            valid_elevations = np.isin(all_factors[..., 0], self.factor_space.factor_vals[0])  # elevation
            valid_azimuths = np.isin(all_factors[..., 1], self.factor_space.factor_vals[1])  # azimuth
            valid_factors_mask = (valid_elevations & valid_azimuths)

            valid_factors = all_factors[valid_factors_mask]
            valid_factors_idxs = self.factor_space.factor2idx(valid_factors)

            dataset[valid_factors_idxs] = data_mesh[valid_factors_mask]

        return dataset

class Shapes3D(FactorisedDataset):
    '''Shapes3D dataset.

    Images of shapes in a room with varying color, shape and viewing angle variational
    factors.  The images are formatted as 64x64 with 3 (RGB) channels.

    The data set was first used in the paper "Disentangling by Factorising"
    (https://arxiv.org/abs/1802.05983).  It is publicly available at:
    https://storage.googleapis.com/3d-shapes/3dshapes.h5
    '''

    _path = "./_data/3dshapes.h5"
    _all_factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    _all_factor_dims = [10, 10, 10, 8, 4, 15]

    def _load_dataset(self):
        '''Load the dataset as an array of observation tensors.

        The index of an observation corresponds to the index of the underlying
        factor configuration in the FactorSpace.
        '''
        def get_index(factors):
            '''
            Find in the index of factors within the loaded array shape of [480000,64,64,3].
            '''
            indices = np.zeros(len(factors))
            base = 1
            for idx in range(len(self._all_factor_dims) - 1, -1, -1):
                indices += factors[..., idx] * base
                base *= self._all_factor_dims[idx]
            return indices

        dataset = np.zeros((len(self.factor_space), 64, 64, 3))

        full_dataset = h5py.File(self._path, 'r')
        images = full_dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
        # labels = full_dataset['labels']  # array shape [480000,6], float64

        factors = self.factor_space.get_all_factors()
        dataset[self.factor_space.factor2idx(factors)] = images[get_index(factors)]

        dataset /= 255  # Normalisation.

        return dataset