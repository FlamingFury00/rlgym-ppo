"""
Retrace Implementation for Muesli Algorithm

This module implements the Retrace algorithm for computing multi-step returns
with importance sampling correction, which is a critical component of Muesli.

Based on "Safe and Efficient Off-policy Reinforcement Learning" (Munos et al., 2016)
and the Muesli paper implementation details.
"""

import torch


class RetraceEstimator:
    """
    Implements Retrace algorithm for safe off-policy learning.

    Retrace provides bias-corrected n-step returns that are safe for off-policy learning
    by using importance sampling with a trace cutting mechanism.
    """

    def __init__(self, gamma=0.99, lambda_retrace=1.0, device=None):
        """
        Initialize Retrace estimator.

        Args:
            gamma (float): Discount factor
            lambda_retrace (float): Retrace lambda parameter (typically 1.0)
            device (torch.device): Device for computations
        """
        self.gamma = gamma
        self.lambda_retrace = lambda_retrace
        self.device = device

    def compute_retrace_targets(
        self, rewards, values, next_values, dones, importance_weights, n_steps=5
    ):
        """
        Compute Retrace targets for value function learning.

        Args:
            rewards (torch.Tensor): Immediate rewards [batch_size, n_steps]
            values (torch.Tensor): Value estimates [batch_size, n_steps]
            next_values (torch.Tensor): Next state values [batch_size, n_steps]
            dones (torch.Tensor): Done flags [batch_size, n_steps]
            importance_weights (torch.Tensor): Importance sampling weights [batch_size, n_steps]
            n_steps (int): Number of steps for n-step returns

        Returns:
            torch.Tensor: Retrace targets [batch_size, n_steps]
        """
        batch_size, sequence_length = rewards.shape

        # Initialize Retrace targets
        retrace_targets = torch.zeros_like(values)

        # Clamp importance weights for stability (key Retrace innovation)
        c_t = torch.clamp(importance_weights, max=1.0)

        # Compute Retrace targets using backward recursion
        for t in reversed(range(sequence_length)):
            if t == sequence_length - 1:
                # Terminal step: use immediate reward + discounted next value
                retrace_targets[:, t] = rewards[:, t] + self.gamma * next_values[
                    :, t
                ] * (1 - dones[:, t])
            else:
                # Recursive Retrace computation
                td_error = (
                    rewards[:, t]
                    + self.gamma * next_values[:, t] * (1 - dones[:, t])
                    - values[:, t]
                )

                retrace_targets[:, t] = values[:, t] + c_t[:, t] * (
                    td_error
                    + self.gamma
                    * self.lambda_retrace
                    * c_t[:, t + 1]
                    * (retrace_targets[:, t + 1] - next_values[:, t])
                )

        return retrace_targets

    def compute_multi_step_returns(self, trajectory_data, policy_model, value_model):
        """
        Compute multi-step returns using Retrace for a trajectory.

        Args:
            trajectory_data (dict): Dictionary containing trajectory information
            policy_model: Current policy model for computing importance weights
            value_model: Value model for value estimates

        Returns:
            dict: Dictionary with Retrace targets and related information
        """
        states = trajectory_data["states"]
        actions = trajectory_data["actions"]
        rewards = trajectory_data["rewards"]
        old_log_probs = trajectory_data["old_log_probs"]
        dones = trajectory_data["dones"]

        batch_size, sequence_length = rewards.shape

        # Compute current policy log probabilities
        with torch.no_grad():
            current_log_probs = []
            values = []

            for t in range(sequence_length):
                step_states = states[:, t]
                step_actions = actions[:, t]

                # Get current policy probabilities
                log_probs, _, step_values, _ = (
                    policy_model.get_action_log_prob_and_entropy(
                        step_states, step_actions
                    )
                )
                current_log_probs.append(log_probs)
                values.append(step_values)

            current_log_probs = torch.stack(current_log_probs, dim=1)
            values = torch.stack(values, dim=1)

            # Compute importance sampling weights
            importance_weights = torch.exp(current_log_probs - old_log_probs)

            # Compute next state values (for bootstrap)
            next_values = torch.zeros_like(values)
            for t in range(sequence_length - 1):
                next_values[:, t] = values[:, t + 1]

            # For last step, use current value as approximation
            next_values[:, -1] = values[:, -1]

        # Compute Retrace targets
        retrace_targets = self.compute_retrace_targets(
            rewards, values, next_values, dones, importance_weights
        )

        # Compute advantages using Retrace targets
        advantages = retrace_targets - values

        return {
            "retrace_targets": retrace_targets,
            "advantages": advantages,
            "importance_weights": importance_weights,
            "values": values,
        }

    def compute_policy_gradient_targets(
        self, trajectory_data, policy_model, dynamics_model, reward_model, value_model
    ):
        """
        Compute policy gradient targets using Retrace with model-based lookahead.

        This implements the full Muesli approach where model-based rollouts
        are combined with Retrace for robust policy gradient estimation.

        Args:
            trajectory_data (dict): Trajectory data
            policy_model: Current policy model
            dynamics_model: Learned dynamics model
            reward_model: Learned reward model
            value_model: Value model

        Returns:
            dict: Policy gradient targets and related information
        """
        states = trajectory_data["states"]
        batch_size, sequence_length = states.shape[:2]

        # Perform model-based rollouts for each timestep
        model_based_returns = []

        with torch.no_grad():
            for t in range(sequence_length):
                step_states = states[:, t]  # s_t from real trajectory

                # Get hidden state representation from current policy's representation network
                current_hidden_state = policy_model.get_hidden_state(step_states)

                # Perform n-step model rollout
                discounted_return = torch.zeros(batch_size, device=states.device)
                rollout_hidden_state = current_hidden_state  # State for model rollout
                discount = 1.0

                for k in range(min(5, sequence_length - t)):  # 5-step lookahead
                    # Select action using current policy for model rollout
                    # Assuming policy_model.get_action returns action suitable for dynamics/reward models
                    # This might need adjustment based on MuesliPolicy's get_action output format
                    # For simplicity, let's assume it provides a suitable action tensor.
                    # Deterministic actions are often used in model rollouts for value estimation.
                    rollout_action, _ = policy_model.get_action(
                        rollout_hidden_state, deterministic=True
                    )

                    # Ensure rollout_action is on the correct device and has appropriate dimensions
                    rollout_action = rollout_action.to(states.device)
                    if (
                        len(rollout_action.shape) == 1 and policy_model.policy_type != 0
                    ):  # Continuous or MultiDiscrete might need unsqueeze
                        if (
                            policy_model.action_space_size > 1
                            or policy_model.policy_type == 2
                        ):  # Cont or MultiDiscrete with >1 dim
                            rollout_action = rollout_action.unsqueeze(-1)
                    elif (
                        len(rollout_action.shape) == 0 and policy_model.policy_type == 0
                    ):  # Discrete action, needs to be (batch, 1)
                        rollout_action = rollout_action.unsqueeze(-1)

                    # Predict reward and next state
                    predicted_reward = reward_model(
                        rollout_hidden_state, rollout_action
                    )
                    next_rollout_hidden_state = dynamics_model(
                        rollout_hidden_state, rollout_action
                    )

                    # Add discounted reward
                    if hasattr(reward_model, "categorical_to_scalar"):
                        reward_scalar = reward_model.categorical_to_scalar(
                            predicted_reward
                        )
                    else:
                        reward_scalar = predicted_reward

                    discounted_return += discount * reward_scalar
                    discount *= self.gamma
                    rollout_hidden_state = next_rollout_hidden_state  # Update state for next step of rollout

                # Add final bootstrap value from the value_model
                final_value = value_model(
                    rollout_hidden_state
                )  # Value of the k-step rollout state
                if hasattr(
                    value_model, "categorical_value_to_scalar"
                ):  # Adapt if value model is categorical
                    final_value = value_model.categorical_value_to_scalar(final_value)
                elif len(final_value.shape) > 1:  # Ensure scalar value per batch item
                    final_value = final_value.squeeze(-1)

                discounted_return += discount * final_value
                model_based_returns.append(discounted_return)

        model_based_returns = torch.stack(model_based_returns, dim=1)

        # Combine with Retrace for robust off-policy correction
        # Pass the policy_model to compute_multi_step_returns for IS weight calculation
        retrace_data = self.compute_multi_step_returns(
            trajectory_data, policy_model, value_model
        )

        # Weighted combination of model-based and Retrace returns
        alpha_model = 0.5  # Weight for model-based component
        combined_targets = (
            alpha_model * model_based_returns
            + (1 - alpha_model) * retrace_data["retrace_targets"]
        )

        return {
            "combined_targets": combined_targets,
            "model_based_returns": model_based_returns,
            "retrace_targets": retrace_data["retrace_targets"],
            "advantages": combined_targets - retrace_data["values"],
            "importance_weights": retrace_data["importance_weights"],
        }


class TrajectoryProcessor:
    """
    Processes trajectories for Retrace computation in Muesli.

    Handles the conversion of raw experience data into the format
    required for Retrace computation.
    """

    def __init__(self, n_step=5, gamma=0.99):
        """
        Initialize trajectory processor.

        Args:
            n_step (int): Number of steps for n-step returns
            gamma (float): Discount factor
        """
        self.n_step = n_step
        self.gamma = gamma

    def process_trajectory(self, trajectory):
        """
        Process a single trajectory for Retrace computation.

        Args:
            trajectory (list): List of (state, action, reward, done, log_prob) tuples

        Returns:
            dict: Processed trajectory data
        """
        states = []
        actions = []
        rewards = []
        dones = []
        old_log_probs = []

        for step_data in trajectory:
            states.append(step_data["state"])
            actions.append(step_data["action"])
            rewards.append(step_data["reward"])
            dones.append(step_data["done"])
            old_log_probs.append(step_data["log_prob"])

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_log_probs = torch.stack(old_log_probs)

        # Create sliding windows for n-step computation
        trajectory_windows = []

        for i in range(len(trajectory) - self.n_step + 1):
            window = {
                "states": states[i : i + self.n_step],
                "actions": actions[i : i + self.n_step],
                "rewards": rewards[i : i + self.n_step],
                "dones": dones[i : i + self.n_step],
                "old_log_probs": old_log_probs[i : i + self.n_step],
            }
            trajectory_windows.append(window)

        return trajectory_windows

    def batch_trajectories(self, trajectory_windows):
        """
        Batch multiple trajectory windows for efficient computation.

        Args:
            trajectory_windows (list): List of trajectory window dictionaries

        Returns:
            dict: Batched trajectory data
        """
        if not trajectory_windows:
            return {}

        # Stack all windows into batches
        batched_data = {}
        for key in trajectory_windows[0].keys():
            batched_data[key] = torch.stack(
                [window[key] for window in trajectory_windows]
            )

        return batched_data
