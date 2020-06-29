import math

import torch
import torch.nn as nn

from src.coders.utils import nnNorm


class EncoderConv(nn.Module):
    '''CNN for encoding observations in a spherical latent space.'''

    def __init__(self,
                 obs_channels=3,
                 obs_width=64,
                 kernels=[[32, 4, 2, 0]] * 2 + [[64, 4, 2, 0]] * 2,  # (out_channels, kernel_size, stride, padding)
                 n_hid=[256],
                 lat_dim=4,
                 activation_fn=nn.ReLU,
                 output_activation=nnNorm,
                 batch_norm=False,
                 device="cpu"):
        '''An encoder neural network to map images to latent vectors.

        The network takes in images of shape (obs_channels, obs_width, obs_width) and maps them to vectors
        of length lat_dim.

        Args:
            obs_channels: The number of channels in the input observation (image).
            obs_width: The width (and height) of the input observation (image).
            kernels: A list of kernels [(number of output channels, kernel size, stride, padding), ...],
                     each kernel corresponding to a convolutional layer to construct.
            n_hid: A list of integers determining the size of the hidden layers in the fully connected
                   readout layers.  Necessarily, the first and last layers will be determined by the number
                   of outputs from the conv layers and the requested output size, respectively, so these do
                   no need to be specified.
            lat_dim: The size of the output layer (i.e. the dimension of the latent space).
            activation_fn:  The activation function to use.
            output_activation: The output activation.  nnNorm, as provided in the utils class, will normalise
                               the output to have unit norm.
            batch_norm:  Whether to apply batch normalisation between conv layers.
            device: The device of the network.  Should be "cpu" or "cuda".
        '''
        super().__init__()

        self.device = device

        conv_layers = []
        _width = obs_width
        in_channels = obs_channels
        for out_channels, size, stride, padding in kernels:
            conv = nn.Conv2d(in_channels, out_channels, size, stride, padding)
            #             conv_layers.append(nn.Sequential(conv, activation_fn()))
            seq = [conv]
            if batch_norm:
                seq.append(nn.BatchNorm2d(out_channels))
            seq.append(activation_fn())
            conv_layers.append(nn.Sequential(*seq))
            in_channels = out_channels
            _width = math.floor((_width + 2 * padding - size) / stride) + 1
        self.conv_layers = nn.Sequential(*conv_layers)

        fc_layers = []
        n_fc = [in_channels * int(_width) ** 2] + n_hid + [lat_dim]
        for idx, (n_in, n_out) in enumerate(zip(n_fc[:-1], n_fc[1:])):
            layer = nn.Linear(n_in, n_out)
            if idx == len(n_fc) - 2:
                act = output_activation
            else:
                act = activation_fn
            fc_layers.append(nn.Sequential(layer, act()))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        if x.dim() == 3:
            # (C,H,W) --> (B=1,C,H,W)
            x.unsqueeze_(0)
        x = self.conv_layers(x.to(self.device))
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x.squeeze()


class DecoderConv(nn.Module):
    '''CNN for mapping latent space vectors to reconstructed images.'''

    def __init__(self,
                 obs_channels=3,
                 obs_width=64,
                 kernels=[[32, 4, 2, 0]] * 2 + [[64, 4, 2, 0]] * 2,  # (out_channels, kernel_size, stride, padding)
                 n_hid=[256],
                 lat_dim=4,
                 activation_fn=nn.ReLU,
                 output_activation=nn.Sigmoid,
                 batch_norm=False,
                 device="cpu"):
        '''An decoder neural network to map latent vectors to images.

        The network takes in vectors with dimention lat_dim and maps them to images of shape
        (obs_channels, obs_width, obs_width).

        Args:
            obs_channels: The number of channels in the input observation (image).
            obs_width: The width (and height) of the input observation (image).
            kernels: A list of kernels [(number of output channels, kernel size, stride, padding), ...],
                     each kernel corresponding to a convolutional layer to construct.
            n_hid: A list of integers determining the size of the hidden layers in the fully connected
                   readout layers.  Necessarily, the first and last layers will be determined by the number
                   of outputs from the conv layers and the requested output size, respectively, so these do
                   no need to be specified.
            lat_dim: The size of the output layer (i.e. the dimension of the latent space).
            activation_fn:  The activation function to use.
            output_activation: The output activation.
            batch_norm:  Whether to apply batch normalisation between conv layers.
            device: The device of the network.  Should be "cpu" or "cuda".
        '''
        super().__init__()

        self.device = device

        _widths = [obs_width]
        for out_channels, size, stride, padding in kernels:
            _widths.append(math.floor((_widths[-1] + 2 * padding - size) / stride) + 1)

        fc_layers = []
        self._conv_inp_channels, self._conv_inp_width = kernels[-1][0], _widths[-1]
        n_fc = [lat_dim] + n_hid + [self._conv_inp_width ** 2 * self._conv_inp_channels]
        for n_in, n_out in zip(n_fc[:-1], n_fc[1:]):
            layer = nn.Linear(n_in, n_out)
            fc_layers.append(nn.Sequential(layer, activation_fn()))

        self.fc_layers = nn.Sequential(*fc_layers)

        out_channels = obs_channels
        kernels = kernels.copy()
        for idx, k in enumerate(kernels.copy()):
            k = k.copy()
            out_channels, k[0] = k[0], out_channels
            kernels[idx] = k.copy()

        conv_layers = []
        in_channels = self._conv_inp_channels
        for idx, (out_channels, size, stride, padding) in enumerate(reversed(kernels), start=1):
            deconv_width = (_widths[-idx] - 1) * stride + size
            #             print("deconv_width", deconv_width, "width", _widths[-idx-1])
            out_padding = _widths[-idx - 1] - deconv_width
            conv = nn.ConvTranspose2d(in_channels, out_channels, size, stride, padding, output_padding=out_padding)
            if idx == len(kernels):
                act = output_activation
            else:
                act = activation_fn
            seq = [conv]
            if batch_norm:
                seq.append(nn.BatchNorm2d(out_channels))
            seq.append(act())
            conv_layers.append(nn.Sequential(*seq))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        if x.dim() == 1:
            # (Z) --> (B=1,Z)
            x.unsqueeze_(0)
        x = self.fc_layers(x.to(self.device))
        x = x.reshape(x.shape[0], self._conv_inp_channels, self._conv_inp_width, self._conv_inp_width)
        x = self.conv_layers(x)
        return x.squeeze()