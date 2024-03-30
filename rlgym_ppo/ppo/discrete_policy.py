"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""

import numpy as np
import torch
import torch.nn as nn

from moe import MoE


class DiscretePolicy(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device
        self.n_actions = n_actions
        self.policy = MoE(
            input_shape,
            n_actions,
            num_experts=8,
            hidden_size=layer_sizes[0],
            noisy_gating=True,
            k=4,
        ).to(device)

    def get_output(self, obs):
        t = type(obs)
        if t != torch.Tensor:
            if t != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        output, _ = self.policy(obs)
        return output

    def get_action(self, obs, deterministic=False):
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """

        probs = self.get_output(obs)
        probs = probs.view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)

        if deterministic:
            return probs.cpu().numpy().argmax(), 0

        action = torch.multinomial(probs, 1, True)
        log_prob = torch.log(probs).gather(-1, action)

        return action.flatten().cpu(), log_prob.flatten().cpu()

    def get_backprop_data(self, obs, acts):
        """
        Function to compute the data necessary for backpropagation.
        :param obs: Observations to pass through the policy.
        :param acts: Actions taken by the policy.
        :return: Action log probs & entropy.
        """
        acts = acts.long()
        output, _ = self.policy(obs)
        probs = output.view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)

        log_probs = torch.log(probs)
        action_log_probs = log_probs.gather(-1, acts)
        entropy = -(log_probs * probs).sum(dim=-1)

        return action_log_probs.to(self.device), entropy.to(self.device).mean()
