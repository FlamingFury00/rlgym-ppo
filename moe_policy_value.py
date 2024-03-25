import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from moe import MoE


class MoEPolicyValue(nn.Module):
    def __init__(
        self, obs_space_size, act_space_size, hidden_size=256, num_experts=10, k=4
    ):
        super(MoEPolicyValue, self).__init__()
        self.obs_space_size = obs_space_size
        self.act_space_size = act_space_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy MoE
        self.policy_moe = MoE(
            input_size=obs_space_size,
            output_size=act_space_size,
            num_experts=num_experts,
            hidden_size=hidden_size,
            k=k,
        ).to(self.device)

        # Value MoE
        self.value_moe = MoE(
            input_size=obs_space_size,
            output_size=1,
            num_experts=num_experts,
            hidden_size=hidden_size,
            k=k,
        ).to(self.device)

    def forward(self, x):
        policy_output, _ = self.policy_moe(x)
        value_output, _ = self.value_moe(x)

        # Convert policy_output tensor to a distribution object
        policy_dist = dist.Categorical(logits=policy_output)

        return policy_dist, value_output.squeeze(-1)

    def get_action(self, obs, deterministic=False):
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """
        t = type(obs)
        if t != torch.Tensor:
            if t != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        probs, _ = self.policy_moe(obs)
        probs = probs.view(-1, self.act_space_size)
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
        probs, _ = self.policy_moe(obs)
        probs = probs.view(-1, self.act_space_size)
        probs = torch.clamp(probs, min=1e-11, max=1)

        log_probs = torch.log(probs)
        action_log_probs = log_probs.gather(-1, acts)
        entropy = -(log_probs * probs).sum(dim=-1)

        return action_log_probs.to(self.device), entropy.to(self.device).mean()
