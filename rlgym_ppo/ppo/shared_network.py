import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class SharedNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.n_actions = n_actions
        self.device = device

        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"

        # Shared layers
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        self.shared_layers = nn.Sequential(*layers).to(self.device)

        # Policy output layers
        self.policy_logits = nn.Linear(prev_size, n_actions).to(self.device)
        self.policy_softmax = nn.Softmax(dim=-1).to(self.device)

        # Value output layer
        self.value_estimator = nn.Linear(prev_size, 1).to(self.device)

    def forward(self, obs):
        # Process observations through shared layers
        shared_output = self.shared_layers(obs)

        # Policy output
        policy_logits = self.policy_logits(shared_output)
        policy_probs = self.policy_softmax(policy_logits)

        # Value output
        value = self.value_estimator(shared_output)

        return policy_probs, value

    def get_output(self, obs):
        t = type(obs)
        if t != torch.Tensor:
            if isinstance(obs, np.ndarray):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            else:
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        policy_probs, value = self.forward(obs)
        return policy_probs, value

    def get_action(self, obs, deterministic=False):
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """
        policy_probs, _ = self.get_output(obs)
        policy_probs = policy_probs.view(-1, self.n_actions)
        policy_probs = torch.clamp(policy_probs, min=1e-11, max=1)

        if deterministic:
            action = policy_probs.argmax(dim=-1)
            log_prob = torch.zeros_like(action)
        else:
            action_dist = Categorical(policy_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

        return action.cpu(), log_prob.cpu()

    def get_backprop_data(self, obs, acts):
        """
        Function to compute the data necessary for backpropagation.
        :param obs: Observations to pass through the policy.
        :param acts: Actions taken by the policy.
        :return: Action log probs & entropy.
        """
        acts = acts.long()
        policy_probs, _ = self.get_output(obs)
        policy_probs = policy_probs.view(-1, self.n_actions)
        policy_probs = torch.clamp(policy_probs, min=1e-11, max=1)

        action_log_probs = torch.log(policy_probs).gather(-1, acts)
        entropy = -(policy_probs * torch.log(policy_probs)).sum(dim=-1).mean()

        return action_log_probs.to(self.device), entropy.to(self.device)
