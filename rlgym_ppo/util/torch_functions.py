"""
File: torch_functions.py
Author: Matthew Allen

Description:
    A helper file for misc. PyTorch functions.

"""

import torch.nn as nn
import torch


class MapContinuousToAction(nn.Module):
    """
    A class for policies using the continuous action space. Continuous policies output N*2 values for N actions where
    each value is in the range [-1, 1]. Half of these values will be used as the mean of a multi-variate normal distribution
    and the other half will be used as the diagonal of the covariance matrix for that distribution. Since variance must
    be positive, this class will map the range [-1, 1] for those values to the desired range (defaults to [0.1, 1]) using
    a simple linear transform.
    """

    def __init__(self, range_min=0.1, range_max=1):
        super().__init__()

        tanh_range = [-1, 1]
        self.m = (range_max - range_min) / (tanh_range[1] - tanh_range[0])
        self.b = range_min - tanh_range[0] * self.m

    def forward(self, x):
        n = x.shape[-1] // 2
        # map the right half of x from [-1, 1] to [range_min, range_max].
        return x[..., :n], x[..., n:] * self.m + self.b


def compute_gae(
    rews, dones, truncated, values, gamma=0.99, lmbda=0.95, return_std=None
):
    """
    Function to estimate the advantage function for a series of states and actions using the
    general advantage estimator (GAE).

    :param rews: List of rewards.
    :param dones: List of done signals.
    :param truncated: List of truncated signals.
    :param values: List of value function estimates.
    :param gamma: Gamma hyper-parameter.
    :param lmbda: Lambda hyper-parameter.
    :param return_std: Standard deviation of the returns (used for reward normalization).
    :return: Bootstrapped value function estimates, GAE results, returns.
    """
    rews_tensor = torch.tensor(rews, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, dtype=torch.float32)
    truncated_tensor = torch.tensor(truncated, dtype=torch.float32)
    values_tensor = torch.tensor(values, dtype=torch.float32)

    if return_std is not None:
        norm_rews_tensor = torch.clamp(rews_tensor / return_std, -10, 10)
    else:
        norm_rews_tensor = rews_tensor

    values_tensor = values_tensor[:-1]
    next_values_tensor = torch.cat(
        (values_tensor[1:], torch.tensor([0.0]))
    )  # Pad last value for calculation

    not_dones = 1 - dones_tensor
    not_truncated = 1 - truncated_tensor

    deltas = norm_rews_tensor + gamma * next_values_tensor * not_dones - values_tensor

    advantages = torch.zeros_like(rews_tensor)
    returns = torch.zeros_like(rews_tensor)
    last_gae_lam = torch.tensor(0.0)
    last_return = torch.tensor(0.0)

    for t in reversed(range(len(rews_tensor))):
        not_done = not_dones[t]
        not_trunc = not_truncated[t]
        delta = deltas[t]

        advantages[t] = last_gae_lam = (
            delta + gamma * lmbda * not_done * not_trunc * last_gae_lam
        )
        returns[t] = last_return = (
            rews_tensor[t] + last_return * gamma * not_done * not_trunc
        )

    value_targets = values_tensor + advantages

    return value_targets, advantages, returns.tolist()


class MultiDiscreteRolv(nn.Module):
    """
    A class to handle the multi-discrete action space in Rocket League. There are 8 potential actions, 5 of which can be
    any of {-1, 0, 1} and 3 of which can be either of {0, 1}. This class takes 21 logits, appends -inf to the final 3
    such that each of the 8 actions has 3 options (to avoid a ragged list), then builds a categorical distribution over
    each class for each action. Credit to Rolv Arild for coming up with this method.
    """

    def __init__(self, bins):
        super().__init__()
        self.distribution = None
        self.bins = bins

    def make_distribution(self, logits):
        """
        Function to make the multi-discrete categorical distribution for a group of logits.
        :param logits: Logits which parameterize the distribution.
        :return: None.
        """

        # Split the 21 logits into the expected bins.
        logits = torch.split(logits, self.bins, dim=-1)

        # Separate triplets from the split logits.
        triplets = torch.stack(logits[:5], dim=-1)

        # Separate duets and pad the final dimension with -inf to create triplets.
        duets = torch.nn.functional.pad(
            torch.stack(logits[5:], dim=-1), pad=(0, 0, 0, 1), value=float("-inf")
        )

        # Un-split the logits now that the duets have been converted into triplets and reshape them into the correct shape.
        logits = torch.cat((triplets, duets), dim=-1).swapdims(-1, -2)

        # Construct a distribution with our fixed logits.
        self.distribution = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        return self.distribution.log_prob(action).sum(dim=-1)

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy().sum(
            dim=-1
        )  # Unsure about this sum operation.
