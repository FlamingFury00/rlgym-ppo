import numpy as np
import torch
import torch.nn as nn


class MapContinuousToAction(nn.Module):
    def __init__(self, range_min=0.1, range_max=1):
        super().__init__()
        self.scale = (range_max - range_min) / 2
        self.bias = (range_max + range_min) / 2

    def forward(self, x):
        n = x.shape[-1] // 2
        return x[..., :n], x[..., n:] * self.scale + self.bias


def compute_gae(
    rewards, dones, truncated, values, gamma=0.99, lambda_=0.95, epsilon=1e-8
):
    """
    Compute Generalized Advantage Estimation (GAE) with improved stability and normalization.

    :param rewards: List of rewards
    :param dones: List of done flags
    :param truncated: List of truncation flags
    :param values: List of value estimates
    :param gamma: Discount factor
    :param lambda_: GAE parameter
    :param epsilon: Small constant for numerical stability
    :return: Tuple of (value_targets, advantages, returns)
    """
    num_steps = len(rewards)
    advantages = np.zeros(num_steps, dtype=np.float32)
    last_gae = 0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = 0  # For the last step, there's no next state
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = (
            delta + gamma * lambda_ * (1 - dones[t]) * (1 - truncated[t]) * last_gae
        )

    # Compute returns
    returns = advantages + np.array(values[:-1])

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + epsilon)

    # Compute value targets with GAE
    value_targets = returns

    return (
        torch.as_tensor(value_targets, dtype=torch.float32),
        torch.as_tensor(advantages, dtype=torch.float32),
        torch.as_tensor(returns, dtype=torch.float32),
    )


class MultiDiscreteRolv(nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.distribution = None
        self.bins = bins
        self.neg_inf = float("-inf")

    def make_distribution(self, logits):
        split_logits = torch.split(logits, self.bins, dim=-1)
        triplets = torch.stack(split_logits[:5], dim=-1)
        duets = torch.nn.functional.pad(
            torch.stack(split_logits[5:], dim=-1), pad=(0, 0, 0, 1), value=self.neg_inf
        )
        combined_logits = torch.cat((triplets, duets), dim=-1).transpose(-1, -2)
        self.distribution = torch.distributions.Categorical(logits=combined_logits)

    def log_prob(self, action):
        return self.distribution.log_prob(action).sum(dim=-1)

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
