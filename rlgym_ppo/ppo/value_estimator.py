"""
File: value_estimator.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which models the value function of a policy.
"""

import numpy as np
import torch
import torch.nn as nn

from moe import MoE


class ValueEstimator(nn.Module):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device
        self.value_net = MoE(
            input_shape,
            1,
            num_experts=8,
            hidden_size=layer_sizes[0],
            noisy_gating=True,
            k=4,
        ).to(device)

    def forward(self, x):
        t = type(x)
        if t != torch.Tensor:
            if t != np.array:
                x = np.asarray(x)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        output, _ = self.value_net(x)
        return output
