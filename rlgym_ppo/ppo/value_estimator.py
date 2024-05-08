"""
File: value_estimator.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which models the value function of a policy.
"""

import numpy as np
import torch
import torch.nn as nn


class ValueEstimator(nn.Module):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device

        assert (
            len(layer_sizes) > 0
        ), "At least one layer must be specified to build the neural network!"

        layers = []
        prev_size = input_shape
        for idx, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, size))

            # Apply Kaiming initialization to the layer
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity="relu")

            if idx < len(layer_sizes) - 1:
                # Batch Normalization
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU(inplace=True))

            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], 1))
        # Apply Kaiming initialization to the final layer
        nn.init.kaiming_normal_(layers[-1].weight, nonlinearity="linear")

        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        """
        Forward pass of the network. Accepts input data and outputs the corresponding prediction.

        Parameters:
        x (torch.Tensor, numpy.ndarray, or list): The input data.

        Returns:
        torch.Tensor: The output from the network.
        """
        # Ensure the input is a torch.Tensor
        if not isinstance(x, torch.Tensor):
            # Convert input to numpy array if it's a list, then to a tensor
            x = np.array(x, dtype=np.float32) if not isinstance(x, np.ndarray) else x
            x = torch.tensor(x, dtype=torch.float32, device=self.model[0].weight.device)

        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        # Perform the forward pass
        return self.model(x)
