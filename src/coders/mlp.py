import torch.nn as nn

from src.coders.utils import nnNorm


class Coder(nn.Module):
    '''A simple MLP network.'''

    def __init__(self, n_in, n_out, n_hid=[32], output_activation=nn.Sigmoid):
        '''A standard MLP.

        An MLP with n_in inputs, n_out outputs and ReLU activations.

        Args:
            n_in: The size of the input layer.
            n_out: The size of the output layer.
            n_hid: A list of sizes for the hidden layer.
            output_activation: The output activation.
        '''
        super().__init__()

        if type(n_hid) != list:
            n_hid = [n_hid]
        n_layers = [n_in] + n_hid + [n_out]

        self.layers = []
        for i_layer, (n1, n2) in enumerate(zip(n_layers, n_layers[1:])):
            mods = [nn.Linear(n1, n2, bias=True)]
            act_fn = nn.ReLU if i_layer < len(n_layers) - 2 else output_activation
            if act_fn is not None:
                mods.append(act_fn())
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(Coder):

    def __init__(self, n_in, n_out, n_hid=[32]):
        '''MLP for decoding unit-norm vectors to vectors with all values between
        zero and one.

        Args:
            n_in: The size of the input layer.
            n_out: The size of the output layer.
            n_hid: A list of sizes for the hidden layer.
        '''
        super().__init__(n_in, n_out, n_hid, output_activation=nn.Sigmoid)

class Encoder(Coder):

    def __init__(self, n_in, n_out, n_hid=[32]):
        '''MLP for encoding vectors with all values between zero and one to
        unit-norm vectors.

        Args:
            n_in: The size of the input layer.
            n_out: The size of the output layer.
            n_hid: A list of sizes for the hidden layer.
        '''
        super().__init__(n_in, n_out, n_hid, output_activation=nnNorm)