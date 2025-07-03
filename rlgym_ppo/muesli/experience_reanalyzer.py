"""
Experience Reanalyzer for Muesli Algorithm

This module implements the experience reanalysis component that re-evaluates
stored experiences with updated models to improve sample efficiency.
"""

import torch
import numpy as np
from collections import deque
import random


class ExperienceReanalyzer:
    """
    Reanalyzes stored experiences using updated models.

    This is a key component of Muesli that allows the algorithm to extract
    more value from stored experiences by re-evaluating them with improved models.
    """

    def __init__(
        self,
        replay_buffer_size=100000,
        reanalysis_ratio=0.5,
        n_step_unroll=5,
        device=None,
        seed=42,
    ):
        """
        Initialize the experience reanalyzer.

        Args:
            replay_buffer_size (int): Maximum size of the replay buffer
            reanalysis_ratio (float): Ratio of reanalyzed to fresh experiences
            n_step_unroll (int): Number of steps for model unrolling
            device (torch.device): Device for computations
            seed (int): Random seed for reproducibility
        """
        self.replay_buffer_size = replay_buffer_size
        self.reanalysis_ratio = reanalysis_ratio
        self.n_step_unroll = n_step_unroll
        self.device = device

        # Initialize replay buffer as a deque for efficient operations
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Set random seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

        # Statistics
        self.total_experiences_stored = 0
        self.total_reanalysis_performed = 0

    def store_experience(
        self, states, actions, rewards, next_states, dones, values, log_probs
    ):
        """
        Store experience in the replay buffer.

        Args:
            states (torch.Tensor): States
            actions (torch.Tensor): Actions taken
            rewards (torch.Tensor): Immediate rewards
            next_states (torch.Tensor): Next states
            dones (torch.Tensor): Done flags
            values (torch.Tensor): Value estimates
            log_probs (torch.Tensor): Action log probabilities
        """
        # Convert to numpy for storage efficiency
        experience = {
            "states": (
                states.cpu().numpy() if isinstance(states, torch.Tensor) else states
            ),
            "actions": (
                actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
            ),
            "rewards": (
                rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
            ),
            "next_states": (
                next_states.cpu().numpy()
                if isinstance(next_states, torch.Tensor)
                else next_states
            ),
            "dones": dones.cpu().numpy() if isinstance(dones, torch.Tensor) else dones,
            "values": (
                values.cpu().numpy() if isinstance(values, torch.Tensor) else values
            ),
            "log_probs": (
                log_probs.cpu().numpy()
                if isinstance(log_probs, torch.Tensor)
                else log_probs
            ),
        }

        self.replay_buffer.append(experience)
        self.total_experiences_stored += 1

    def sample_experiences_for_reanalysis(self, batch_size):
        """
        Sample experiences from replay buffer for reanalysis.

        Args:
            batch_size (int): Number of experiences to sample

        Returns:
            list: Sampled experiences
        """
        if len(self.replay_buffer) == 0:
            return []

        # Sample without replacement
        sample_size = min(batch_size, len(self.replay_buffer))
        sampled_indices = self.rng.choice(
            len(self.replay_buffer), size=sample_size, replace=False
        )

        sampled_experiences = [self.replay_buffer[i] for i in sampled_indices]
        return sampled_experiences

    def reanalyze_experiences(
        self,
        experiences,
        dynamics_model,
        reward_model,
        value_model,
        policy_model,
        gamma=0.99,
    ):
        """
        Reanalyze experiences using updated models.

        Args:
            experiences (list): List of experience dictionaries
            dynamics_model: Updated dynamics model
            reward_model: Updated reward model
            value_model: Updated value model
            policy_model: Updated policy model
            gamma (float): Discount factor

        Returns:
            dict: Reanalyzed experience data
        """
        if not experiences:
            return {}

        # Convert experiences back to tensors
        batch_states = torch.tensor(
            [exp["states"] for exp in experiences],
            dtype=torch.float32,
            device=self.device,
        )
        batch_actions = torch.tensor(
            [exp["actions"] for exp in experiences],
            dtype=torch.float32,
            device=self.device,
        )
        batch_rewards = torch.tensor(
            [exp["rewards"] for exp in experiences],
            dtype=torch.float32,
            device=self.device,
        )

        batch_size = len(experiences)

        # Recompute hidden states using current representation model
        with torch.no_grad():
            hidden_states = policy_model.get_hidden_state(batch_states)

        # Perform n-step unroll using updated models
        reanalyzed_values = self._compute_n_step_values(
            hidden_states,
            batch_actions,
            batch_rewards,
            dynamics_model,
            reward_model,
            value_model,
            gamma,
        )

        # Recompute advantages using new value estimates
        reanalyzed_advantages = self._compute_advantages(
            batch_rewards, reanalyzed_values, gamma
        )

        # Get updated policy log probabilities
        current_log_probs, _, _, _ = policy_model.get_action_log_prob_and_entropy(
            batch_states, batch_actions
        )

        self.total_reanalysis_performed += batch_size

        return {
            "states": batch_states,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "values": reanalyzed_values,
            "advantages": reanalyzed_advantages,
            "log_probs": current_log_probs,
            "hidden_states": hidden_states,
        }

    def _compute_n_step_values(
        self,
        hidden_states,
        actions,
        rewards,
        dynamics_model,
        reward_model,
        value_model,
        gamma,
    ):
        """
        Compute n-step value estimates using model rollouts.

        Args:
            hidden_states (torch.Tensor): Initial hidden states
            actions (torch.Tensor): Actions for rollout
            rewards (torch.Tensor): Immediate rewards
            dynamics_model: Dynamics model for state prediction
            reward_model: Reward model for reward prediction
            value_model: Value model for final value estimation
            gamma (float): Discount factor

        Returns:
            torch.Tensor: N-step value estimates
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Initialize value computation
        n_step_values = torch.zeros(batch_size, device=device)
        current_states = hidden_states
        discount = 1.0

        # Unroll for n steps
        for step in range(
            min(self.n_step_unroll, actions.size(1) if len(actions.shape) > 1 else 1)
        ):
            # Get action for this step
            if len(actions.shape) > 1:
                step_action = actions[:, step]
            else:
                step_action = actions

            # Add immediate reward
            if len(rewards.shape) > 1:
                step_reward = rewards[:, step]
            else:
                step_reward = rewards

            n_step_values += discount * step_reward

            # Predict next state and reward using models
            if step < self.n_step_unroll - 1:
                with torch.no_grad():
                    # Predict next hidden state
                    current_states = dynamics_model(current_states, step_action)

                    # Update discount
                    discount *= gamma

        # Add final state value estimate
        with torch.no_grad():
            final_values = value_model(current_states)
            if hasattr(value_model, "categorical_value_to_scalar"):
                final_values = value_model.categorical_value_to_scalar(final_values)
            elif len(final_values.shape) > 1:
                final_values = final_values.squeeze(-1)

            n_step_values += discount * final_values

        return n_step_values

    def _compute_advantages(self, rewards, values, gamma, gae_lambda=0.95):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards (torch.Tensor): Rewards
            values (torch.Tensor): Value estimates
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter

        Returns:
            torch.Tensor: Advantage estimates
        """
        # Simple advantage computation for now
        # This could be enhanced with proper GAE computation
        advantages = rewards - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def get_mixed_batch(self, fresh_experiences, batch_size):
        """
        Create a mixed batch of fresh and reanalyzed experiences.

        Args:
            fresh_experiences: Fresh experience data
            batch_size (int): Target batch size

        Returns:
            dict: Mixed batch of experiences
        """
        # Calculate sizes for fresh vs reanalyzed experiences
        reanalyzed_size = int(batch_size * self.reanalysis_ratio)
        fresh_size = batch_size - reanalyzed_size

        # Sample and reanalyze experiences if replay buffer has data
        if len(self.replay_buffer) > 0 and reanalyzed_size > 0:
            # Note: reanalysis would be performed by the caller with current models
            # This method just prepares the mixed batch structure
            pass

        # Combine fresh and reanalyzed data
        # Implementation depends on the structure of fresh_experiences
        mixed_batch = {
            "fresh_size": fresh_size,
            "reanalyzed_size": reanalyzed_size,
            "fresh_experiences": fresh_experiences,
            "needs_reanalysis": len(self.replay_buffer) > 0 and reanalyzed_size > 0,
        }

        return mixed_batch

    def get_statistics(self):
        """
        Get reanalysis statistics.

        Returns:
            dict: Statistics about the reanalyzer
        """
        return {
            "replay_buffer_size": len(self.replay_buffer),
            "max_buffer_size": self.replay_buffer_size,
            "total_experiences_stored": self.total_experiences_stored,
            "total_reanalysis_performed": self.total_reanalysis_performed,
            "reanalysis_ratio": self.reanalysis_ratio,
            "buffer_utilization": len(self.replay_buffer) / self.replay_buffer_size,
        }

    def clear_buffer(self):
        """Clear the replay buffer."""
        self.replay_buffer.clear()

    def set_reanalysis_ratio(self, new_ratio):
        """
        Update the reanalysis ratio.

        Args:
            new_ratio (float): New ratio between 0 and 1
        """
        assert 0.0 <= new_ratio <= 1.0, "Reanalysis ratio must be between 0 and 1"
        self.reanalysis_ratio = new_ratio
        print(f"Updated reanalysis ratio to {new_ratio}")


class MuesliExperienceBuffer:
    """
    Enhanced experience buffer for Muesli that supports both fresh experiences
    and reanalyzed experiences from the replay buffer.
    """

    def __init__(self, max_size, device, reanalyzer=None):
        """
        Initialize the Muesli experience buffer.

        Args:
            max_size (int): Maximum buffer size
            device (torch.device): Device for computations
            reanalyzer (ExperienceReanalyzer): Optional reanalyzer instance
        """
        self.max_size = max_size
        self.device = device
        self.reanalyzer = reanalyzer

        # Storage for current episode experiences
        self.current_experiences = []

    def add_experience(self, state, action, reward, next_state, done, value, log_prob):
        """
        Add experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state
            done: Done flag
            value: Value estimate
            log_prob: Action log probability
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "value": value,
            "log_prob": log_prob,
        }

        self.current_experiences.append(experience)

        # Store in reanalyzer if available
        if self.reanalyzer is not None:
            self.reanalyzer.store_experience(
                state, action, reward, next_state, done, value, log_prob
            )

    def get_batch_data(self, include_reanalyzed=True):
        """
        Get batch data including fresh and optionally reanalyzed experiences.

        Args:
            include_reanalyzed (bool): Whether to include reanalyzed experiences

        Returns:
            dict: Batch data ready for training
        """
        if not self.current_experiences:
            return {}

        # Convert current experiences to tensors
        batch_data = self._experiences_to_tensors(self.current_experiences)

        # Add reanalyzed experiences if requested and available
        if include_reanalyzed and self.reanalyzer is not None:
            mixed_batch = self.reanalyzer.get_mixed_batch(
                batch_data, len(self.current_experiences)
            )
            batch_data["mixed_batch_info"] = mixed_batch

        return batch_data

    def _experiences_to_tensors(self, experiences):
        """Convert list of experiences to tensor format."""
        states = torch.stack([exp["state"] for exp in experiences])
        actions = torch.stack([exp["action"] for exp in experiences])
        rewards = torch.stack([exp["reward"] for exp in experiences])
        next_states = torch.stack([exp["next_state"] for exp in experiences])
        dones = torch.stack([exp["done"] for exp in experiences])
        values = torch.stack([exp["value"] for exp in experiences])
        log_probs = torch.stack([exp["log_prob"] for exp in experiences])

        return {
            "states": states.to(self.device),
            "actions": actions.to(self.device),
            "rewards": rewards.to(self.device),
            "next_states": next_states.to(self.device),
            "dones": dones.to(self.device),
            "values": values.to(self.device),
            "log_probs": log_probs.to(self.device),
        }

    def clear(self):
        """Clear current experiences."""
        self.current_experiences.clear()
