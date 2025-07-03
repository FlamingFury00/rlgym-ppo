"""
Muesli Policy for Enhanced Policy Optimization

This module implements the enhanced policy network for Muesli that combines
regularized policy optimization with model learning capabilities.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from rlgym_ppo.util import torch_functions


class ObservationStacker:
    """
    Observation stacker for Muesli algorithm to stack multiple frames.

    This provides temporal context by maintaining a sliding window of observations.
    The stacking approach is important for temporal understanding and matches the reference
    implementations that use 8-frame stacking.
    """

    def __init__(self, obs_size, stack_size=8):
        """
        Initialize observation stacker.

        Args:
            obs_size (int): Size of single observation
            stack_size (int): Number of observations to stack (default: 8)
        """
        self.obs_size = obs_size
        self.stack_size = stack_size
        self.stacked_obs = np.zeros((stack_size, obs_size), dtype=np.float32)

    def add_observation(self, obs):
        """
        Add a new observation and return stacked observations.

        Args:
            obs (np.ndarray): New observation

        Returns:
            np.ndarray: Stacked observations
        """
        # Roll stack and add new observation
        self.stacked_obs = np.roll(self.stacked_obs, -1, axis=0)
        self.stacked_obs[-1] = obs

        # Return flattened stacked observations
        return self.stacked_obs.flatten()

    def reset(self, initial_obs=None):
        """
        Reset the observation stack.

        Args:
            initial_obs (np.ndarray, optional): Initial observation to fill stack
        """
        if initial_obs is not None:
            # Fill stack with initial observation
            for i in range(self.stack_size):
                self.stacked_obs[i] = initial_obs
        else:
            # Clear stack
            self.stacked_obs.fill(0.0)

    def get_stacked_size(self):
        """Get the size of stacked observations."""
        return self.obs_size * self.stack_size


class MuesliPolicy(nn.Module):
    """
    Enhanced policy network for Muesli algorithm.

    This policy extends the standard PPO policy with additional capabilities
    needed for Muesli's regularized policy optimization and model learning.
    """

    def __init__(
        self,
        obs_space_size,
        action_space_size,
        policy_type,
        layer_sizes,
        device,
        continuous_var_range=(0.1, 1.0),
        hidden_state_size=256,
    ):
        """
        Initialize the Muesli policy.

        Args:
            obs_space_size (int): Size of observation space
            action_space_size (int): Size of action space
            policy_type (int): Type of policy (0=discrete, 1=multi-discrete, 2=continuous)
            layer_sizes (list): List of layer sizes for the network
            device (torch.device): Device to run computations on
            continuous_var_range (tuple): Range for continuous action variance
            hidden_state_size (int): Size of hidden state representation
        """
        super(MuesliPolicy, self).__init__()

        self.device = device
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.policy_type = policy_type
        self.hidden_state_size = hidden_state_size

        # Build representation network (observation -> hidden state)
        self.representation_net = self._build_representation_network(
            obs_space_size, hidden_state_size, layer_sizes
        )

        # Build policy head (hidden state -> action distribution)
        if policy_type == 2:  # Continuous
            self.policy_head = self._build_continuous_policy_head(
                hidden_state_size, action_space_size, continuous_var_range
            )
        elif policy_type == 1:  # Multi-discrete
            self.policy_head = self._build_multi_discrete_policy_head(
                hidden_state_size, action_space_size
            )
        else:  # Discrete
            self.policy_head = self._build_discrete_policy_head(
                hidden_state_size, action_space_size
            )
            # Add softmax for efficient discrete action handling
            self.softmax = nn.Softmax(dim=-1)

        # Value head (hidden state -> value estimate)
        self.value_head = self._build_value_head(hidden_state_size)

        # Initialize weights
        self._initialize_weights()

    def _build_representation_network(self, input_size, output_size, layer_sizes):
        """Build the representation network."""
        layers = []
        prev_size = input_size

        for size in layer_sizes:
            layers.extend([nn.Linear(prev_size, size), nn.ReLU()])
            prev_size = size

        layers.extend(
            [
                nn.Linear(prev_size, output_size),
                nn.Tanh(),  # Keep hidden states in [-1, 1] range
            ]
        )

        return nn.Sequential(*layers)

    def _build_continuous_policy_head(self, input_size, action_size, var_range):
        """Build continuous policy head using efficient approach like PPO."""
        # Output mean and std for each action
        policy_net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, action_size * 2),
            nn.Tanh(),
        )

        # Store variance range for mapping
        self.var_min, self.var_max = var_range
        # Create affine mapping like PPO
        self.affine_map = torch_functions.MapContinuousToAction(
            range_min=var_range[0], range_max=var_range[1]
        )

        return policy_net

    def _build_discrete_policy_head(self, input_size, action_size):
        """Build discrete policy head using efficient approach like PPO."""
        return nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, action_size),
        )

    def _build_multi_discrete_policy_head(self, input_size, action_sizes):
        """Build multi-discrete policy head using efficient approach."""
        if isinstance(action_sizes, int):
            # Single discrete action space
            return nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, action_sizes),
            )
        else:
            # Multi-discrete action space (like RocketLeague) - use efficient approach
            bins = [3, 3, 3, 3, 3, 2, 2, 2]  # Default RL bins
            n_output_nodes = sum(bins)

            policy_net = nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, n_output_nodes),
            )

            # Store multi-discrete handler
            self.multi_discrete = torch_functions.MultiDiscreteRolv(bins)
            self.bins = bins

            return policy_net

    def _build_value_head(self, input_size):
        """Build value estimation head."""
        return nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 1),
        )

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def get_hidden_state(self, obs):
        """
        Convert observation to hidden state representation.

        Args:
            obs (torch.Tensor): Observations

        Returns:
            torch.Tensor: Hidden state representations
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        obs = obs.to(self.device)
        return self.representation_net(obs)

    def get_output(self, obs):
        """
        Get policy output for different action types.

        Args:
            obs (torch.Tensor): Observations

        Returns:
            torch.Tensor: Action probabilities for discrete policy,
                         (mean, std) for continuous policy,
                         logits for multi-discrete policy
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.to(self.device)

        hidden_state = self.get_hidden_state(obs)

        if self.policy_type == 0:  # Discrete
            logits = self.policy_head(hidden_state)
            return self.softmax(logits)
        elif self.policy_type == 2:  # Continuous
            policy_output = self.policy_head(hidden_state)
            return self.affine_map(policy_output)
        else:  # Multi-discrete or other
            return self.policy_head(hidden_state)

    def get_action_and_value(self, obs, deterministic=False):
        """
        Get action and value estimate from observation.

        Args:
            obs (torch.Tensor): Observations
            deterministic (bool): Whether to sample deterministically

        Returns:
            tuple: (action, log_prob, value, hidden_state)
        """
        # Get hidden state representation
        hidden_state = self.get_hidden_state(obs)

        # Get value estimate
        value = self.value_head(hidden_state).squeeze(-1)

        # Get action distribution and sample
        if self.policy_type == 2:  # Continuous
            action, log_prob = self._sample_continuous_action(
                hidden_state, deterministic
            )
        elif self.policy_type == 1:  # Multi-discrete
            action, log_prob = self._sample_multi_discrete_action(
                hidden_state, deterministic
            )
        else:  # Discrete
            action, log_prob = self._sample_discrete_action(hidden_state, deterministic)

        return action, log_prob, value, hidden_state

    def _sample_continuous_action(self, hidden_state, deterministic=False):
        """Sample from continuous action distribution using efficient approach like PPO."""
        policy_output = self.policy_head(hidden_state)
        mean, std = self.affine_map(policy_output)

        if deterministic:
            # Return mean for deterministic action, log_prob = 0
            action = torch.clamp(mean, -1.0, 1.0)
            log_prob = torch.zeros(
                action.shape[0], dtype=torch.float32, device=self.device
            )
            return action.cpu(), log_prob.cpu()

        # Create distribution and sample
        dist = Normal(mean, std)
        action = dist.sample().clamp(min=-1, max=1)

        # Use custom log probability computation like PPO
        log_prob = self._compute_continuous_logpdf(action, mean, std)

        # Sum log probabilities across action dimensions
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)
        else:
            log_prob = log_prob.sum()

        return action.cpu(), log_prob.cpu()

    def _compute_continuous_logpdf(self, x, mean, std):
        """
        Compute log probability density function like PPO continuous policy.
        """
        msq = mean * mean
        ssq = std * std
        xsq = x * x

        term1 = -torch.divide(msq, (2 * ssq))
        term2 = torch.divide(mean * x, ssq)
        term3 = -torch.divide(xsq, (2 * ssq))
        term4 = torch.log(1 / torch.sqrt(2 * np.pi * ssq))

        return term1 + term2 + term3 + term4

    def _sample_multi_discrete_action(self, hidden_state, deterministic=False):
        """Sample from multi-discrete action distribution using efficient approach like PPO."""
        if hasattr(self, "multi_discrete"):
            # Use efficient multi-discrete sampling for RL-like environments
            logits = self.policy_head(hidden_state)

            if deterministic:
                # Deterministic action selection like PPO multi-discrete
                start = 0
                actions = []
                for split in self.bins:
                    action_logits = logits[..., start : start + split]
                    action = action_logits.argmax(dim=-1)
                    actions.append(action)
                    start += split
                action = torch.stack(actions, dim=-1)
                log_prob = torch.zeros(
                    action.shape[0], dtype=torch.float32, device=self.device
                )
                return action, log_prob
            else:
                # Stochastic sampling using efficient multi-discrete
                self.multi_discrete.make_distribution(logits)
                action = self.multi_discrete.sample()
                log_prob = self.multi_discrete.log_prob(action)

                return action, log_prob

        else:
            # Fallback to single discrete action space using efficient method
            logits = self.policy_head(hidden_state)
            probs = self.softmax(logits)
            probs = probs.view(-1, self.action_space_size)
            probs = torch.clamp(probs, min=1e-11, max=1)

            if deterministic:
                action = probs.argmax(dim=-1)
                log_prob = torch.zeros_like(action, dtype=torch.float32)
            else:
                action = torch.multinomial(probs, 1, True)
                log_prob = torch.log(probs).gather(1, action)
                action = action.flatten()
                log_prob = log_prob.flatten()

        return action, log_prob

    def _sample_discrete_action(self, hidden_state, deterministic=False):
        """Sample from discrete action distribution using efficient direct computation."""
        logits = self.policy_head(hidden_state)
        probs = self.softmax(logits)
        probs = probs.view(-1, self.action_space_size)
        probs = torch.clamp(probs, min=1e-11, max=1)

        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = torch.zeros_like(action, dtype=torch.float32)
        else:
            action = torch.multinomial(probs, 1, True)
            log_prob = torch.log(probs).gather(1, action)
            action = action.flatten()
            log_prob = log_prob.flatten()

        return action, log_prob

    def get_action_log_prob_and_entropy(self, obs, actions):
        """
        Get log probabilities and entropy for given observations and actions.

        Args:
            obs (torch.Tensor): Observations
            actions (torch.Tensor): Actions taken

        Returns:
            tuple: (log_probs, entropy, values, hidden_states)
        """
        # Get hidden state representation
        hidden_state = self.get_hidden_state(obs)

        # Get value estimate
        values = self.value_head(hidden_state).squeeze(-1)

        # Compute log probabilities and entropy using efficient methods
        if self.policy_type == 2:  # Continuous
            # Use efficient continuous computation
            policy_output = self.policy_head(hidden_state)
            mean, std = self.affine_map(policy_output)

            # Compute log probabilities using custom method like PPO
            log_prob = self._compute_continuous_logpdf(actions, mean, std)

            # Sum log probabilities across action dimensions
            if len(log_prob.shape) > 1:
                log_probs = log_prob.sum(dim=-1)
            else:
                log_probs = log_prob.sum()

            # Create distribution for entropy computation
            dist = Normal(mean, std)
            entropy = dist.entropy().sum(dim=-1).mean()

        elif self.policy_type == 1:  # Multi-discrete
            if hasattr(self, "multi_discrete"):
                # Use efficient multi-discrete computation
                logits = self.policy_head(hidden_state)
                self.multi_discrete.make_distribution(logits)

                log_probs = self.multi_discrete.log_prob(actions)
                entropy = self.multi_discrete.entropy().mean()
            else:
                # Fallback to efficient discrete computation
                logits = self.policy_head(hidden_state)
                probs = self.softmax(logits)
                probs = probs.view(-1, self.action_space_size)
                probs = torch.clamp(probs, min=1e-11, max=1)

                # Convert actions to long tensor for indexing and ensure proper shape
                actions = actions.long().to(self.device)
                if len(actions.shape) == 1:
                    actions = actions.view(-1, 1)
                elif len(actions.shape) > 2:
                    actions = actions.view(-1, 1)

                # Compute log probabilities directly
                log_probs = torch.log(probs).gather(1, actions).squeeze(-1)

                # Compute entropy directly
                entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

        else:  # Discrete
            # Use efficient discrete computation
            logits = self.policy_head(hidden_state)
            probs = self.softmax(logits)
            probs = probs.view(-1, self.action_space_size)
            probs = torch.clamp(probs, min=1e-11, max=1)

            # Convert actions to long tensor for indexing and ensure proper shape
            actions = actions.long().to(self.device)
            if len(actions.shape) == 1:
                actions = actions.view(-1, 1)
            elif len(actions.shape) > 2:
                actions = actions.view(-1, 1)

            # Compute log probabilities directly
            log_probs = torch.log(probs).gather(1, actions).squeeze(-1)

            # Compute entropy directly
            entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

        return log_probs, entropy, values, hidden_state

    def get_backprop_data(self, obs, actions):
        """
        Get log probabilities and entropy for backpropagation.

        This method provides compatibility with the PPO interface and uses
        efficient approaches for each action type.

        Args:
            obs (torch.Tensor): Observations
            actions (torch.Tensor): Actions taken

        Returns:
            tuple: (log_probs, entropy)
        """
        if self.policy_type == 0:  # Discrete - use efficient approach
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs = obs.to(self.device)
            actions = actions.long().to(self.device)

            # Get probabilities directly
            probs = self.get_output(obs)
            probs = probs.view(-1, self.action_space_size)
            probs = torch.clamp(probs, min=1e-11, max=1)

            # Ensure actions have proper shape for gathering
            if len(actions.shape) == 1:
                actions = actions.view(-1, 1)
            elif len(actions.shape) > 2:
                actions = actions.view(-1, 1)

            # Compute log probabilities directly
            log_probs = torch.log(probs).gather(1, actions).squeeze(-1)

            # Compute entropy directly
            entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

            return log_probs, entropy

        elif self.policy_type == 1:  # Multi-discrete - use efficient approach like PPO
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs = obs.to(self.device)

            if hasattr(self, "multi_discrete"):
                # Use efficient multi-discrete computation
                logits = self.get_output(obs)
                self.multi_discrete.make_distribution(logits)

                log_probs = self.multi_discrete.log_prob(actions).to(self.device)
                entropy = self.multi_discrete.entropy().to(self.device).mean()

                return log_probs, entropy
            else:
                # Fallback to general method
                log_probs, entropy, _, _ = self.get_action_log_prob_and_entropy(
                    obs, actions
                )
                return log_probs, entropy

        else:  # Continuous - use efficient approach like PPO
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs = obs.to(self.device)

            mean, std = self.get_output(obs)

            # Compute log probabilities using custom method
            log_prob = self._compute_continuous_logpdf(actions, mean, std)

            # Sum log probabilities across action dimensions
            if len(log_prob.shape) > 1:
                log_probs = log_prob.sum(dim=-1)
            else:
                log_probs = log_prob.sum()

            # Create distribution for entropy
            dist = Normal(mean, std)
            entropy = dist.entropy().sum(dim=-1).mean()

            return log_probs, entropy

    def get_action(self, obs, deterministic=False):
        """
        Get actions for inference.

        This method provides compatibility with the agent manager interface.
        For discrete actions, it uses the efficient approach like PPO.

        Args:
            obs (torch.Tensor): Observations
            deterministic (bool): Whether to sample deterministically

        Returns:
            tuple: (actions, log_probs)
        """
        if self.policy_type == 0:  # Discrete - use efficient approach
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs = obs.to(self.device)

            # Get probabilities directly
            probs = self.get_output(obs)
            probs = probs.view(-1, self.action_space_size)
            probs = torch.clamp(probs, min=1e-11, max=1)

            if deterministic:
                action = probs.argmax(dim=-1).cpu()
                log_prob = torch.zeros_like(action)
            else:
                action = torch.multinomial(probs, 1, True)
                log_prob = torch.log(probs).gather(1, action)
                action = action.flatten().cpu()
                log_prob = log_prob.flatten().cpu()

            return action, log_prob
        else:
            # For other action types, use the general method
            actions, log_probs, _, _ = self.get_action_and_value(obs, deterministic)
            return actions, log_probs

    def _compute_continuous_log_prob_entropy(self, hidden_state, actions):
        """Compute log probabilities and entropy for continuous actions using efficient approach like PPO."""
        policy_output = self.policy_head(hidden_state)
        mean, std = self.affine_map(policy_output)

        # Compute log probabilities using custom method like PPO
        log_prob = self._compute_continuous_logpdf(actions, mean, std)

        # Sum log probabilities across action dimensions
        if len(log_prob.shape) > 1:
            log_probs = log_prob.sum(dim=-1)
        else:
            log_probs = log_prob.sum()

        # Create distribution for entropy computation
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1).mean()

        return log_probs, entropy

    def _compute_multi_discrete_log_prob_entropy(self, hidden_state, actions):
        """Compute log probabilities and entropy for multi-discrete actions using efficient approach."""
        if hasattr(self, "multi_discrete"):
            # Use efficient multi-discrete computation
            logits = self.policy_head(hidden_state)
            self.multi_discrete.make_distribution(logits)

            log_probs = self.multi_discrete.log_prob(actions)
            entropy = self.multi_discrete.entropy().mean()

        else:
            # Fallback to single discrete action space
            logits = self.policy_head(hidden_state)
            probs = self.softmax(logits)
            probs = probs.view(-1, self.action_space_size)
            probs = torch.clamp(probs, min=1e-11, max=1)

            # Convert actions to long tensor for indexing and ensure proper shape
            actions = actions.long().to(self.device)
            if len(actions.shape) == 1:
                actions = actions.view(-1, 1)
            elif len(actions.shape) > 2:
                actions = actions.view(-1, 1)

            # Compute log probabilities directly
            log_probs = torch.log(probs).gather(1, actions).squeeze(-1)

            # Compute entropy directly
            entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

        return log_probs, entropy

    def _compute_discrete_log_prob_entropy(self, hidden_state, actions):
        """Compute log probabilities and entropy for discrete actions using efficient direct computation."""
        logits = self.policy_head(hidden_state)
        probs = self.softmax(logits)
        probs = probs.view(-1, self.action_space_size)
        probs = torch.clamp(probs, min=1e-11, max=1)

        # Convert actions to long tensor for indexing and ensure proper shape
        actions = actions.long().to(self.device)
        if len(actions.shape) == 1:
            actions = actions.view(-1, 1)
        elif len(actions.shape) > 2:
            actions = actions.view(-1, 1)

        # Compute log probabilities directly
        log_probs = torch.log(probs).gather(1, actions).squeeze(-1)

        # Compute entropy directly
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

        return log_probs, entropy

    def get_conservative_policy_loss(
        self,
        obs,
        actions,
        old_log_probs,
        advantages,
        clip_range=0.2,
        conservative_weight=1.0,
    ):
        """
        Compute Conservative Multi-step Policy Optimization (CMPO) loss.

        This implements the full CMPO loss from the Muesli paper including:
        - Standard importance sampling with clipping (first term)
        - Conservative regularization term (second term) using one-step lookahead

        Args:
            obs (torch.Tensor): Observations
            actions (torch.Tensor): Actions taken
            old_log_probs (torch.Tensor): Log probabilities from old policy
            advantages (torch.Tensor): Advantage estimates
            clip_range (float): PPO clipping range
            conservative_weight (float): Weight for conservative regularization

        Returns:
            tuple: (policy_loss, kl_divergence)
        """
        # Get current policy log probabilities and hidden states
        current_log_probs, entropy, _, hidden_states = (
            self.get_action_log_prob_and_entropy(obs, actions)
        )

        # Compute importance sampling ratio
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Normalize advantages (critical for CMPO)
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # First term: Standard clipped importance sampling objective
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        first_term = -torch.min(
            ratio * normalized_advantages, clipped_ratio * normalized_advantages
        ).mean()

        # Second term: Conservative regularization using one-step lookahead
        # This is the key innovation of CMPO that prevents policy degradation
        second_term = self._compute_conservative_regularization_term(
            hidden_states, obs, normalized_advantages, conservative_weight
        )

        # Combined CMPO loss
        total_policy_loss = first_term + second_term

        # Compute KL divergence for monitoring
        kl_divergence = (
            torch.exp(old_log_probs) * (old_log_probs - current_log_probs)
        ).mean()

        return total_policy_loss, kl_divergence

    def _compute_conservative_regularization_term(
        self, hidden_states, obs, advantages, weight
    ):
        """
        Compute the conservative regularization term (second term of CMPO).

        This implements the full Conservative Multi-step Policy Optimization regularization
        as described in the Muesli paper, using importance sampling and policy lookahead.

        Args:
            hidden_states (torch.Tensor): Current hidden states
            obs (torch.Tensor): Observations
            advantages (torch.Tensor): Normalized advantages
            weight (float): Conservative weight

        Returns:
            torch.Tensor: Conservative regularization loss
        """

        with torch.no_grad():
            if self.policy_type == 0:  # Discrete actions
                logits = self.policy_head(hidden_states)
                probs = self.softmax(logits)
                probs = torch.clamp(probs, min=1e-11, max=1)

                # Use efficient entropy computation
                entropy = -(probs * torch.log(probs)).sum(dim=-1)
                regularization_term = entropy.mean()

            elif self.policy_type == 1:  # Multi-discrete actions
                if hasattr(self, "multi_discrete"):
                    # Use efficient multi-discrete entropy computation
                    logits = self.policy_head(hidden_states)
                    self.multi_discrete.make_distribution(logits)
                    entropy = self.multi_discrete.entropy()
                    regularization_term = entropy.mean()
                else:
                    # Fallback to discrete case
                    logits = self.policy_head(hidden_states)
                    probs = self.softmax(logits)
                    probs = torch.clamp(probs, min=1e-11, max=1)
                    entropy = -(probs * torch.log(probs)).sum(dim=-1)
                    regularization_term = entropy.mean()

            else:  # Continuous actions
                # For continuous actions, use policy gradient variance regularization
                policy_output = self.policy_head(hidden_states)
                mean, std = self.affine_map(policy_output)

                # Regularize based on policy uncertainty and advantage variance
                policy_variance = (std * std).mean()
                advantage_variance = advantages.var()

                regularization_term = weight * (policy_variance + advantage_variance)

        return weight * regularization_term
