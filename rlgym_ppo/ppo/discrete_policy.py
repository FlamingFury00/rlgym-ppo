"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An optimized implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device

        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        self.model = nn.Sequential(*layers).to(self.device)
        self.softmax = nn.Softmax(dim=-1)
        self.n_actions = n_actions

    def get_output(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.softmax(self.model(obs))

    def get_action(self, obs, deterministic=False):
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """
        logits = self.get_output(obs)
        if deterministic:
            action = logits.argmax(dim=-1).cpu()
            log_prob = torch.zeros_like(action)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.flatten().cpu(), log_prob.flatten().cpu()

    def get_backprop_data(self, obs, acts):
        """
        Function to compute the data necessary for backpropagation.
        :param obs: Observations to pass through the policy.
        :param acts: Actions taken by the policy.
        :return: Action log probs & entropy.
        """
        acts = acts.long().to(self.device)
        logits = self.get_output(obs)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(acts)
        entropy = dist.entropy().mean()

        return action_log_probs, entropy
