"""
Muesli Learner - Main Algorithm Implementation

This module implements the core Muesli learner that combines regularized policy
optimization with model learning as described in the DeepMind paper.
"""

import os
import time
import torch
import torch.nn as nn

from .dynamics_model import DynamicsModel
from .reward_model import RewardModel, CombinedPredictionHead
from .target_networks import TargetNetworkManager, TargetValueEstimator
from .muesli_policy import MuesliPolicy
from .experience_reanalyzer import ExperienceReanalyzer
from .retrace import RetraceEstimator


class MuesliLearner:
    """
    Main Muesli learner that implements the complete algorithm.

    Combines policy optimization with model learning using multiple loss functions:
    - Policy gradient loss with Conservative Multi-step Policy Optimization (CMPO)
    - Value function loss (L_v)
    - Model learning loss (L_m) for dynamics
    - Reward prediction loss (L_r)
    """

    def __init__(
        self,
        obs_space_size,
        action_space_size,
        policy_type,
        policy_layer_sizes,
        critic_layer_sizes,
        model_layer_sizes,
        continuous_var_range,
        batch_size,
        n_epochs,
        policy_lr,
        critic_lr,
        model_lr,
        clip_range,
        ent_coef,
        mini_batch_size,
        device,
        hidden_state_size=256,
        n_step_unroll=5,
        target_update_rate=0.005,
        model_loss_weight=1.0,
        reward_loss_weight=1.0,
        conservative_weight=1.0,
        reanalysis_ratio=0.0, # Default to no reanalysis
        use_categorical_value=False,
        use_categorical_reward=False,
    ):
        """
        Initialize the Muesli learner.

        Args:
            obs_space_size (int): Observation space size
            action_space_size (int): Action space size
            policy_type (int): Policy type (0=discrete, 1=multi-discrete, 2=continuous)
            policy_layer_sizes (list): Layer sizes for policy network
            critic_layer_sizes (list): Layer sizes for critic network
            model_layer_sizes (list): Layer sizes for model networks
            continuous_var_range (tuple): Variance range for continuous actions
            batch_size (int): Batch size for training
            n_epochs (int): Number of training epochs
            policy_lr (float): Policy learning rate
            critic_lr (float): Critic learning rate
            model_lr (float): Model learning rate
            clip_range (float): PPO clipping range
            ent_coef (float): Entropy coefficient
            mini_batch_size (int): Mini-batch size
            device (torch.device): Device for computations
            hidden_state_size (int): Size of hidden state representation
            n_step_unroll (int): Number of steps for model unrolling
            target_update_rate (float): Target network update rate (tau)
            model_loss_weight (float): Weight for model loss
            reward_loss_weight (float): Weight for reward loss
            conservative_weight (float): Weight for conservative policy updates
            reanalysis_ratio (float): Ratio of reanalyzed experiences to mix in.
            use_categorical_value (bool): Use categorical value representation
            use_categorical_reward (bool): Use categorical reward representation
        """
        self.device = device
        self.reanalysis_ratio = reanalysis_ratio
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.policy_type = policy_type
        self.hidden_state_size = hidden_state_size
        self.n_step_unroll = n_step_unroll
        self.sequential_learning_active = False  # Will be set by learner.py when experience buffer has sequential data

        # Training parameters
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.ent_coef = ent_coef

        # Loss weights
        self.model_loss_weight = model_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.conservative_weight = conservative_weight

        # Validation
        assert (
            batch_size % mini_batch_size == 0
        ), "Batch size must be divisible by mini-batch size"

        # Initialize networks
        self._initialize_networks(
            policy_layer_sizes,
            critic_layer_sizes,
            model_layer_sizes,
            continuous_var_range,
            use_categorical_value,
            use_categorical_reward,
        )

        # Initialize optimizers
        self._initialize_optimizers(policy_lr, critic_lr, model_lr)

        # Initialize target networks
        self.target_manager = TargetNetworkManager(
            tau=target_update_rate, device=device
        )
        self._setup_target_networks()

        # Initialize target value estimator
        self.target_value_estimator = TargetValueEstimator(self.target_manager)

        # Initialize Retrace estimator for robust off-policy learning
        self.retrace_estimator = RetraceEstimator(gamma=0.99, device=device)

        # For compatibility with main learner - expose value network interface
        self.value_net = self.policy.value_head

        # Training statistics
        self.cumulative_model_updates = 0

        # Normalized advantage tracking (as in reference implementations)
        self.advantage_var = 0.0
        self.advantage_beta = 0.99  # Exponential moving average coefficient
        self.advantage_beta_product = 1.0  # Track beta product for bias correction

        # Multi-step advantage variance tracking for model learning
        self.model_advantage_vars = [0.0 for _ in range(self.n_step_unroll)]
        self.model_advantage_betas = [1.0 for _ in range(self.n_step_unroll)]

        self.total_reanalyzed_samples_trained = 0


        self._print_network_info()

    def _initialize_networks(
        self,
        policy_layer_sizes,
        critic_layer_sizes,
        model_layer_sizes,
        continuous_var_range,
        use_categorical_value,
        use_categorical_reward,
    ):
        """Initialize all neural networks."""

        # Main policy network (includes representation, policy, and value heads)
        self.policy = MuesliPolicy(
            self.obs_space_size,
            self.action_space_size,
            self.policy_type,
            policy_layer_sizes,
            self.device,
            continuous_var_range,
            self.hidden_state_size,
        ).to(self.device)

        # For dynamics and reward models, use appropriate action size
        # Discrete actions: use 1 (action index), others: use full action space size
        model_action_size = 1 if self.policy_type == 0 else self.action_space_size

        # Dynamics model for state transitions
        self.dynamics_model = DynamicsModel(
            self.hidden_state_size,
            model_action_size,
            model_layer_sizes,
            self.device,
        ).to(self.device)

        # Reward model for reward prediction
        self.reward_model = RewardModel(
            self.hidden_state_size,
            model_action_size,
            model_layer_sizes,
            self.device,
            categorical=use_categorical_reward,
        ).to(self.device)

        # Combined prediction head (alternative architecture)
        self.prediction_head = CombinedPredictionHead(
            self.hidden_state_size,
            self.action_space_size,
            critic_layer_sizes,
            self.device,
            categorical_value=use_categorical_value,
        ).to(self.device)

    def _initialize_optimizers(self, policy_lr, critic_lr, model_lr):
        """Initialize optimizers for all networks."""

        # Policy optimizer (includes representation and policy heads)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        # Model optimizers with gradient scaling
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics_model.parameters(), lr=model_lr
        )
        self.reward_optimizer = torch.optim.Adam(
            self.reward_model.parameters(), lr=model_lr
        )

        # Prediction head optimizer
        self.prediction_optimizer = torch.optim.Adam(
            self.prediction_head.parameters(), lr=critic_lr
        )

    def _setup_target_networks(self):
        """Setup target networks for stable training."""

        # Register networks with target manager
        self.target_manager.register_network("policy", self.policy)
        self.target_manager.register_network("dynamics", self.dynamics_model)
        self.target_manager.register_network("reward", self.reward_model)
        self.target_manager.register_network("value", self.prediction_head)

        # Perform initial hard update to synchronize target networks
        self.target_manager.hard_update_all_target_networks()

    def _print_network_info(self):
        """Print information about network parameters."""

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        policy_params = count_parameters(self.policy)
        dynamics_params = count_parameters(self.dynamics_model)
        reward_params = count_parameters(self.reward_model)
        prediction_params = count_parameters(self.prediction_head)
        total_params = (
            policy_params + dynamics_params + reward_params + prediction_params
        )

        print("Muesli Network Parameters:")
        print(f"{'Component':<15} {'Count':<10}")
        print("-" * 25)
        print(f"{'Policy':<15} {policy_params:<10}")
        print(f"{'Dynamics':<15} {dynamics_params:<10}")
        print(f"{'Reward':<15} {reward_params:<10}")
        print(f"{'Prediction':<15} {prediction_params:<10}")
        print("-" * 25)
        print(f"{'Total':<15} {total_params:<10}")

    def learn(self, exp_buffer):
        """
        Main learning loop for Muesli algorithm.

        Args:
            exp_buffer: Experience buffer containing training data

        Returns:
            dict: Training metrics and statistics
        """
        n_iterations = 0
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "model_loss": 0.0,
            "reward_loss": 0.0,
            "total_loss": 0.0,
            "entropy": 0.0,
            "kl_divergence": 0.0,
            "clip_fraction": 0.0,
        }

        # Store parameter states for magnitude computation
        policy_before = torch.nn.utils.parameters_to_vector(
            list(self.policy.representation_net.parameters())
            + list(self.policy.policy_head.parameters())
        ).cpu()
        value_before = torch.nn.utils.parameters_to_vector(
            list(self.policy.representation_net.parameters())
            + list(self.policy.value_head.parameters())
        ).cpu()

        start_time = time.time()

        # Training loop
        for epoch in range(self.n_epochs):
            # Check if experience buffer supports sequential data AND has sequences available
            if (
                hasattr(exp_buffer, "enable_sequential")
                and exp_buffer.enable_sequential
                and hasattr(exp_buffer, "get_sequential_batches")
                and hasattr(exp_buffer, "completed_sequences")
                and exp_buffer.completed_sequences is not None
                and len(exp_buffer.completed_sequences) > 0
                and self.n_step_unroll > 1
            ):
                # Use sequential multi-step learning when available
                try:
                    batch_iterator = exp_buffer.get_sequential_batches(
                        self.mini_batch_size, self.n_step_unroll
                    )
                    sequential_learning = True
                except (AttributeError, NotImplementedError, TypeError):
                    # Fallback to regular batches
                    sequential_learning = False
                    batch_iterator = self._get_regular_batch_iterator(exp_buffer)
            else:
                # Use regular single-step learning
                sequential_learning = False
                batch_iterator = self._get_regular_batch_iterator(exp_buffer)
                # print("Using regular batch learning")  # Comment out to reduce spam

            for fresh_batch_data in batch_iterator:
                # fresh_batch_data is a tuple: (actions, log_probs, states, values, advantages, [rewards, next_states])

                final_train_batch = list(fresh_batch_data) # Convert to list to modify

                if self.reanalysis_ratio > 0 and hasattr(exp_buffer, '_sample_for_reanalysis'):
                    # Determine number of reanalyzed samples based on the fresh batch size and ratio
                    # Note: fresh_batch_data[0] is actions, its length gives batch size
                    current_fresh_batch_size = fresh_batch_data[0].size(0)

                    # Calculate how many reanalyzed samples to aim for in this combined batch
                    # The total minibatch size is self.mini_batch_size.
                    # If current_fresh_batch_size is less than self.mini_batch_size,
                    # it implies this might be the last, smaller batch from an epoch.

                    # Let's aim for reanalyzed_samples such that reanalyzed_samples / (fresh_samples + reanalyzed_samples) = ratio
                    # reanalyzed_samples = ratio * fresh_samples / (1 - ratio)
                    num_reanalyzed_to_sample = int( (self.reanalysis_ratio * current_fresh_batch_size) / (1.0 - self.reanalysis_ratio + 1e-9) )

                    if num_reanalyzed_to_sample > 0 :
                        old_experiences = exp_buffer._sample_for_reanalysis(num_reanalyzed_to_sample)
                        if old_experiences:
                            reanalyzed_data_tuple = self._reanalyze_batch_internal(old_experiences)
                            if reanalyzed_data_tuple:
                                self.total_reanalyzed_samples_trained += reanalyzed_data_tuple[0].size(0)
                                # Concatenate fresh and reanalyzed data for each component of the batch
                                # Batch tuple: (actions, log_probs, states, target_values, advantages, rewards, next_states)
                                # The reanalyzed_data_tuple has this format.
                                # The fresh_batch_data might have 5 or 7 elements.

                                for i in range(len(reanalyzed_data_tuple)):
                                    if i < len(final_train_batch):
                                        final_train_batch[i] = torch.cat((final_train_batch[i], reanalyzed_data_tuple[i]), dim=0)
                                    else: # Should not happen if fresh_batch_data has Muesli format (7 elements)
                                        final_train_batch.append(reanalyzed_data_tuple[i])

                current_batch_tuple = tuple(final_train_batch)

                if sequential_learning:
                    # _train_on_sequential_batch expects a dict if it's a true sequential batch from buffer,
                    # or a tuple if it's just the first step.
                    # For now, reanalysis mixing is primarily designed for non-sequential batches.
                    # Handling mixing for true sequential batches would be more complex.
                    # So, if sequential_learning is True, we might just use the original fresh_batch_data
                    # or ensure current_batch_tuple is correctly formatted if it's just the first step.
                    # The current `_train_on_sequential_batch` takes the first step as a tuple.
                    if isinstance(fresh_batch_data, dict) : # True sequential batch from buffer
                         # TODO: Decide how to handle reanalysis for true multi-step sequence batches.
                         # For now, pass original. This means reanalysis only applies to non-sequential part.
                         batch_metrics = self._train_on_sequential_batch(fresh_batch_data)
                    else: # It's the first step of a sequence, processed like a regular batch
                         batch_metrics = self._train_on_sequential_batch(current_batch_tuple)

                else: # Not sequential_learning
                    batch_metrics = self._train_on_batch(current_batch_tuple)

                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in metrics:
                        metrics[key] += value

                n_iterations += 1

        # Update target networks
        self.target_manager.update_all_target_networks()

        # Compute final metrics
        if n_iterations > 0:
            for key in metrics:
                metrics[key] /= n_iterations

        # Compute parameter update magnitude
        policy_after = torch.nn.utils.parameters_to_vector(
            list(self.policy.representation_net.parameters())
            + list(self.policy.policy_head.parameters())
        ).cpu()
        value_after = torch.nn.utils.parameters_to_vector(
            list(self.policy.representation_net.parameters())
            + list(self.policy.value_head.parameters())
        ).cpu()

        policy_update_magnitude = (policy_before - policy_after).norm().item()
        value_update_magnitude = (value_before - value_after).norm().item()

        # Add additional metrics
        additional_metrics = {
            "training_time": (
                (time.time() - start_time) / n_iterations if n_iterations > 0 else 0
            ),
            "cumulative_updates": self.cumulative_model_updates,
            "policy_update_magnitude": policy_update_magnitude,
            "value_update_magnitude": value_update_magnitude,
            "target_network_tau": self.target_manager.tau,
            # Muesli configuration metrics
            "model_loss_weight": self.model_loss_weight,
            "reward_loss_weight": self.reward_loss_weight,
            "conservative_weight": self.conservative_weight,
            "n_step_unroll": self.n_step_unroll,
            "hidden_state_size": self.hidden_state_size,
            # PPO-compatible metric names for reporting
            "PPO Batch Consumption Time": (
                (time.time() - start_time) / n_iterations if n_iterations > 0 else 0
            ),
            "Cumulative Model Updates": self.cumulative_model_updates,
            "Policy Entropy": metrics.get("entropy", 0.0),
            "Mean KL Divergence": metrics.get("kl_divergence", 0.0),
            "Value Function Loss": metrics.get("value_loss", 0.0),
            "SB3 Clip Fraction": metrics.get("clip_fraction", 0.0),
            "Policy Update Magnitude": policy_update_magnitude,
            "Value Function Update Magnitude": value_update_magnitude,
            # Muesli-specific metrics
            "Model Loss": metrics.get("model_loss", 0.0),
            "Reward Loss": metrics.get("reward_loss", 0.0),
            "Total Loss": metrics.get("total_loss", 0.0),
            "Policy Loss": metrics.get("policy_loss", 0.0),
            "Reanalyzed Samples Trained": self.total_reanalyzed_samples_trained,
            "Reanalysis Ratio Setting": self.reanalysis_ratio,
        }

        if hasattr(exp_buffer, 'replay_buffer') and exp_buffer.replay_buffer is not None:
            additional_metrics["Replay Buffer Size (Reanalysis)"] = len(exp_buffer.replay_buffer)
            if hasattr(exp_buffer, 'total_experiences_stored_for_reanalysis'):
                 additional_metrics["Total Stored for Reanalysis"] = exp_buffer.total_experiences_stored_for_reanalysis
            if hasattr(exp_buffer, 'total_reanalysis_samples_provided'):
                 additional_metrics["Total Sampled for Reanalysis"] = exp_buffer.total_reanalysis_samples_provided


        for key, value in additional_metrics.items():
            metrics[key] = value

        self.cumulative_model_updates += n_iterations

        return metrics

    def _recompute_n_step_values_internal(
        self,
        initial_hidden_states: torch.Tensor,
        policy_model: MuesliPolicy,
        dynamics_model: DynamicsModel,
        reward_model: RewardModel,
        value_model: nn.Module, # Can be prediction_head or policy.value_head
        n_step_unroll: int,
        gamma: float
    ) -> torch.Tensor:
        """
        Recomputes n-step values via model-based rollouts from initial_hidden_states.
        Uses current policy for actions, and current dynamics/reward models.
        """
        batch_size = initial_hidden_states.size(0)
        device = initial_hidden_states.device

        n_step_values = torch.zeros(batch_size, device=device)
        current_rollout_hidden_state = initial_hidden_states
        current_discount = 1.0

        with torch.no_grad(): # Rollouts are for target computation, no gradients through them
            for k in range(n_step_unroll):
                # Select action using current policy
                # .get_action might return action on CPU, ensure it's on correct device
                action_obj, _ = policy_model.get_action(current_rollout_hidden_state, deterministic=True)
                rollout_action = action_obj.to(device)

                # Reshape action if necessary (based on policy type)
                if len(rollout_action.shape) == 1 and policy_model.policy_type != 0: # Continuous or MultiDiscrete might need unsqueeze
                    if policy_model.action_space_size > 1 or policy_model.policy_type == 2:
                            rollout_action = rollout_action.unsqueeze(-1)
                elif len(rollout_action.shape) == 0 and policy_model.policy_type == 0: # Discrete action
                            rollout_action = rollout_action.unsqueeze(-1).float() # Ensure float for model
                elif policy_model.policy_type == 0: # Discrete action, ensure float
                    rollout_action = rollout_action.float()


                # Predict reward and next state using current models
                predicted_reward_dist = reward_model(current_rollout_hidden_state, rollout_action)
                if hasattr(reward_model, "categorical_to_scalar"):
                    predicted_reward_scalar = reward_model.categorical_to_scalar(predicted_reward_dist)
                else:
                    predicted_reward_scalar = predicted_reward_dist # Assumes scalar output

                next_rollout_hidden_state = dynamics_model(current_rollout_hidden_state, rollout_action)

                n_step_values += current_discount * predicted_reward_scalar
                current_rollout_hidden_state = next_rollout_hidden_state
                current_discount *= gamma

            # Bootstrap with the final state value from the value_model
            final_value_dist = value_model(current_rollout_hidden_state)
            if hasattr(value_model, "categorical_to_scalar"): # Check if value_model is categorical
                final_value_scalar = value_model.categorical_to_scalar(final_value_dist)
            elif hasattr(value_model, "value_head") and hasattr(value_model.value_head, "categorical_to_scalar"): # If it's MuesliPolicy
                 final_value_scalar = value_model.value_head.categorical_to_scalar(final_value_dist)
            else: # Assumes scalar output
                final_value_scalar = final_value_dist.squeeze(-1) if len(final_value_dist.shape) > 1 else final_value_dist


            n_step_values += current_discount * final_value_scalar

        return n_step_values

    def _reanalyze_batch_internal(self, experience_batch: list[dict]):
        """
        Reanalyzes a batch of old experiences using current models.
        Args:
            experience_batch: A list of experience dictionaries from the replay buffer.
        Returns:
            A tuple (states, actions, new_log_probs, new_target_values, new_advantages, rewards, next_states)
            ready for training, or None if input is empty.
        """
        if not experience_batch:
            return None

        # Convert list of dicts to dict of lists, then to tensors
        # This matches the structure of data stored by MuesliExperienceBuffer.submit_experience
        # Keys: "state", "action", "log_prob", "reward", "next_state", "done", "value"

        states_np = np.array([exp["state"] for exp in experience_batch])
        actions_np = np.array([exp["action"] for exp in experience_batch])
        old_log_probs_np = np.array([exp["log_prob"] for exp in experience_batch])
        rewards_np = np.array([exp["reward"] for exp in experience_batch])
        next_states_np = np.array([exp["next_state"] for exp in experience_batch])
        dones_np = np.array([exp["done"] for exp in experience_batch])
        # old_values_np = np.array([exp["value"] for exp in experience_batch]) # Original value

        # Convert to tensors
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
        # old_log_probs = torch.as_tensor(old_log_probs_np, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
        # dones = torch.as_tensor(dones_np, dtype=torch.bool, device=self.device)
        # old_values = torch.as_tensor(old_values_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # 1. Recompute hidden states using current policy's representation model
            # Assuming states are raw observations
            hidden_states = self.policy.get_hidden_state(states)

            # 2. Recompute n-step target values using model rollouts
            # Using self.prediction_head as the value_model for hidden states
            new_target_values = self._recompute_n_step_values_internal(
                initial_hidden_states=hidden_states,
                policy_model=self.policy,
                dynamics_model=self.dynamics_model,
                reward_model=self.reward_model,
                value_model=self.prediction_head,
                n_step_unroll=self.n_step_unroll,
                gamma=0.99 # Assuming gamma, could be self.gamma if defined for PPO part
            )

            # 3. Recompute advantages
            #    For simplicity, using recomputed target values and current value prediction from policy's value head
            #    (as this is what PPO loss would typically use for current values)
            _, _, current_values_for_advantage, _ = self.policy.get_action_log_prob_and_entropy(states, actions)
            # Ensure current_values_for_advantage is correctly shaped like new_target_values
            if len(current_values_for_advantage.shape) == 0 or current_values_for_advantage.shape[0] != new_target_values.shape[0]:
                 current_values_for_advantage = current_values_for_advantage.unsqueeze(-1) if len(current_values_for_advantage.shape) == 0 else current_values_for_advantage
                 current_values_for_advantage = current_values_for_advantage.expand_as(new_target_values)


            new_advantages = new_target_values - current_values_for_advantage.detach()
            # Normalize advantages (optional, but often done)
            new_advantages = (new_advantages - new_advantages.mean()) / (new_advantages.std() + 1e-8)


            # 4. Get updated log probabilities from the current policy
            new_log_probs, _, _, _ = self.policy.get_action_log_prob_and_entropy(states, actions)

        # Return in the same format as _get_muesli_samples from the buffer for training
        # (actions, log_probs, states, values (targets), advantages, rewards, next_states)
        # Note: 'rewards' and 'next_states' here are the original historical ones, not re-predicted.
        # This is because the policy loss uses them for context but doesn't learn from them directly.
        return actions, new_log_probs, states, new_target_values, new_advantages, rewards, next_states


    def _train_on_batch(self, batch):
        """
        Train on a single mini-batch.

        Args:
            batch: Mini-batch of experience data

        Returns:
            dict: Batch training metrics
        """
        # Update batch unpacking to handle Muesli data
        if len(batch) == 7:  # Muesli batch with rewards and next_states
            # Order from MuesliExperienceBuffer._get_muesli_samples:
            # actions, log_probs, states, values, advantages, rewards, next_states
            (
                batch_acts,
                batch_old_probs,
                batch_obs,
                batch_target_values,
                batch_advantages,
                batch_rewards,
                batch_next_states,
            ) = batch
            # Model learning is now enabled
            model_learning_enabled = True
        else:  # Regular PPO batch (fallback)
            (
                batch_acts,
                batch_old_probs,
                batch_obs,
                batch_target_values,
                batch_advantages,
            ) = batch
            # Create dummy data for compatibility
            batch_rewards = torch.zeros_like(batch_target_values)
            batch_next_states = batch_obs
            model_learning_enabled = False

        # Move data to device
        batch_obs = batch_obs.to(self.device)
        batch_acts = batch_acts.view(batch_acts.size(0), -1).to(self.device)
        batch_old_probs = batch_old_probs.to(self.device)
        batch_target_values = batch_target_values.to(self.device)
        batch_advantages = batch_advantages.to(self.device)

        # Move model learning data to device (if enabled)
        if model_learning_enabled:
            batch_rewards = batch_rewards.to(self.device)
            batch_next_states = batch_next_states.to(self.device)
            # Ensure next_states has proper batch dimension structure
            if len(batch_next_states.shape) == 1:
                # If flattened, reshape to match batch_obs structure
                batch_next_states = batch_next_states.view(batch_obs.shape[0], -1)
            elif batch_next_states.shape[0] != batch_obs.shape[0]:
                # If batch size doesn't match, reshape appropriately
                batch_next_states = batch_next_states.view(batch_obs.shape[0], -1)

        # Zero gradients
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.dynamics_optimizer.zero_grad(set_to_none=True)
        self.reward_optimizer.zero_grad(set_to_none=True)
        self.prediction_optimizer.zero_grad(set_to_none=True)

        # Forward pass through policy
        log_probs, entropy, values, hidden_states = (
            self.policy.get_action_log_prob_and_entropy(batch_obs, batch_acts)
        )

        # Compute Conservative Multi-step Policy Optimization (CMPO) loss
        policy_loss, kl_divergence = self.policy.get_conservative_policy_loss(
            batch_obs,
            batch_acts,
            batch_old_probs,
            batch_advantages,
            self.clip_range,
            self.conservative_weight,
        )

        # Compute value loss
        value_loss = nn.MSELoss()(values, batch_target_values)

        # Compute model losses
        if model_learning_enabled:
            model_loss, reward_loss = self._compute_model_losses(
                {
                    "states": batch_obs,
                    "actions": batch_acts,
                    "rewards": batch_rewards,
                    "next_states": batch_next_states,
                }
            )
        else:
            # Fallback for regular PPO data
            if not hasattr(self, "_model_learning_warning_shown"):
                print(
                    "WARNING: Model learning disabled - using regular PPO experience buffer"
                )
                self._model_learning_warning_shown = True
            model_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            reward_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Combine losses
        total_policy_loss = policy_loss - entropy * self.ent_coef
        total_loss = (
            total_policy_loss
            + value_loss
            + self.model_loss_weight * model_loss
            + self.reward_loss_weight * reward_loss
        )

        # Backward pass
        total_loss.backward()

        # Apply gradient hooks for dynamics network (scale by 0.5 as in reference implementations)
        for param in self.dynamics_model.parameters():
            if param.grad is not None:
                param.grad.data *= 0.5

        # Gradient clipping by value [-1, 1] as in reference implementations
        # Only clip if parameters have gradients
        policy_params_with_grad = [
            p for p in self.policy.parameters() if p.grad is not None
        ]
        dynamics_params_with_grad = [
            p for p in self.dynamics_model.parameters() if p.grad is not None
        ]
        reward_params_with_grad = [
            p for p in self.reward_model.parameters() if p.grad is not None
        ]
        prediction_params_with_grad = [
            p for p in self.prediction_head.parameters() if p.grad is not None
        ]

        if policy_params_with_grad:
            torch.nn.utils.clip_grad_value_(policy_params_with_grad, clip_value=1.0)
        if dynamics_params_with_grad:
            torch.nn.utils.clip_grad_value_(dynamics_params_with_grad, clip_value=1.0)
        if reward_params_with_grad:
            torch.nn.utils.clip_grad_value_(reward_params_with_grad, clip_value=1.0)
        if prediction_params_with_grad:
            torch.nn.utils.clip_grad_value_(prediction_params_with_grad, clip_value=1.0)

        # Optimizer steps
        self.policy_optimizer.step()
        self.dynamics_optimizer.step()
        self.reward_optimizer.step()
        self.prediction_optimizer.step()

        # Compute metrics
        with torch.no_grad():
            ratio = torch.exp(log_probs - batch_old_probs)
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1) > self.clip_range).float()
            ).item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "model_loss": (
                float(model_loss)
                if isinstance(model_loss, (int, float))
                else model_loss.item()
            ),
            "reward_loss": (
                float(reward_loss)
                if isinstance(reward_loss, (int, float))
                else reward_loss.item()
            ),
            "total_loss": total_loss.item(),
            "entropy": entropy.item(),
            "kl_divergence": kl_divergence.item(),
            "clip_fraction": clip_fraction,
        }

    def _compute_model_losses(self, batch_data):
        """
        Compute model learning losses with proper trajectory unrolling.

        Args:
            batch_data (dict): Batch containing states, actions, rewards, next_states, etc.

        Returns:
            tuple: (dynamics_loss, reward_loss)
        """
        states = batch_data["states"]
        actions = batch_data["actions"]
        rewards = batch_data.get("rewards", torch.zeros_like(states[:, 0]))
        next_states = batch_data.get("next_states", states)  # Fallback to same states

        # Ensure all tensors are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # Validate tensor shapes to prevent dimension mismatch errors
        batch_size = states.shape[0]

        # Ensure next_states has the correct batch dimension
        if next_states.shape[0] != batch_size:
            if next_states.numel() == batch_size * states.shape[1]:
                # Reshape if flattened
                next_states = next_states.view(batch_size, -1)
            else:
                # If shapes are incompatible, use states as fallback
                print(
                    f"WARNING: next_states shape {next_states.shape} incompatible with batch_size {batch_size}, using states as fallback"
                )
                next_states = states

        total_dynamics_loss = 0.0
        total_reward_loss = 0.0
        valid_steps = 0

        # Get initial hidden states from current policy representation
        current_hidden_states = self.policy.get_hidden_state(states)

        # This function is called for individual transitions (or the first step of a sequence).
        # For model learning from these, we perform a 1-step prediction.
        # Multi-step model learning from true sequences is handled by _compute_sequential_model_losses.
        max_steps = 1

        # Note: Sequential data provides true environment transitions which are handled elsewhere.
        # This method focuses on learning from individual s,a,r,s' samples.

        for step in range(max_steps):
            # For single-step case, use the entire action vector
            step_actions = actions

            # Handle different reward tensor shapes
            if len(rewards.shape) > 1 and rewards.size(1) > step:
                step_rewards = rewards[:, step]
            elif len(rewards.shape) == 1:
                step_rewards = rewards
            else:
                step_rewards = torch.zeros_like(current_hidden_states[:, 0])

            # Predict next hidden state and reward
            predicted_next_hidden_states = self.dynamics_model(
                current_hidden_states, step_actions
            )
            predicted_rewards = self.reward_model(current_hidden_states, step_actions)

            # For step 0, use actual next observations from experience buffer
            # For subsequent steps, use predicted states (model rollout)
            if step == 0:
                target_next_observations = next_states
                with torch.no_grad():
                    target_next_hidden_states = self.policy.get_hidden_state(
                        target_next_observations
                    )
            else:
                # For multi-step, use the predicted hidden state as target
                # This implements the iterative model learning approach
                target_next_hidden_states = predicted_next_hidden_states.detach()

            # Compute dynamics loss (MSE between predicted and target hidden states)
            dynamics_loss = nn.MSELoss()(
                predicted_next_hidden_states, target_next_hidden_states
            )
            total_dynamics_loss += dynamics_loss
            valid_steps += 1

            # Compute reward loss
            reward_loss = self.reward_model.compute_loss(
                predicted_rewards, step_rewards
            )
            total_reward_loss += reward_loss

            # Update current hidden state for next iteration (model rollout)
            current_hidden_states = predicted_next_hidden_states.detach()

            # For multi-step, we need to generate next actions
            # Use current policy to predict next action from predicted state
            if step < max_steps - 1:
                with torch.no_grad():
                    # Convert hidden state back to observation space for action prediction
                    # This is a simplified approach - in practice, you'd need proper state reconstruction
                    next_action_logits = self.policy.policy_head(current_hidden_states)
                    if self.policy_type == 0:  # Discrete
                        next_actions = (
                            torch.softmax(next_action_logits, dim=-1)
                            .argmax(dim=-1, keepdim=True)
                            .float()
                        )
                    elif self.policy_type == 2:  # Continuous
                        mean, _ = self.policy.affine_map(next_action_logits)
                        next_actions = mean
                    else:  # Multi-discrete
                        if hasattr(self.policy, "multi_discrete"):
                            self.policy.multi_discrete.make_distribution(
                                next_action_logits
                            )
                            next_actions = self.policy.multi_discrete.sample().float()
                        else:
                            next_actions = (
                                torch.softmax(next_action_logits, dim=-1)
                                .argmax(dim=-1, keepdim=True)
                                .float()
                            )

                    # Use predicted action for next step
                    actions = next_actions

        # Average losses over valid steps
        if valid_steps > 0:
            avg_dynamics_loss = total_dynamics_loss / valid_steps
        else:
            avg_dynamics_loss = torch.tensor(
                0.0, device=states.device, requires_grad=True
            )

        avg_reward_loss = (
            total_reward_loss / max_steps
            if max_steps > 0
            else torch.tensor(0.0, device=states.device, requires_grad=True)
        )

        return avg_dynamics_loss, avg_reward_loss

    def _compute_sequential_model_losses(self, sequential_batch_data):
        """
        Compute model learning losses using true sequential data.

        This method handles actual multi-step sequences from the environment,
        providing more accurate model learning than simulated rollouts.

        Args:
            sequential_batch_data (dict): Sequential batch containing:
                - states: (batch_size, sequence_length, obs_size)
                - actions: (batch_size, sequence_length, action_size)
                - rewards: (batch_size, sequence_length)
                - next_states: (batch_size, sequence_length, obs_size)

        Returns:
            tuple: (dynamics_loss, reward_loss)
        """
        states_seq = sequential_batch_data["states"]  # (B, T, obs_size)
        actions_seq = sequential_batch_data["actions"]  # (B, T, action_size)
        rewards_seq = sequential_batch_data["rewards"]  # (B, T)
        next_states_seq = sequential_batch_data["next_states"]  # (B, T, obs_size)

        batch_size, seq_length = states_seq.shape[:2]

        total_dynamics_loss = 0.0
        total_reward_loss = 0.0
        valid_steps = 0

        # Get initial hidden states
        initial_states = states_seq[:, 0]  # (B, obs_size)
        current_hidden_states = self.policy.get_hidden_state(initial_states)

        # Process each step in the sequence
        for step in range(seq_length):
            step_actions = actions_seq[:, step]  # (B, action_size)
            step_rewards = rewards_seq[:, step]  # (B,)
            step_next_states = next_states_seq[:, step]  # (B, obs_size)

            # Predict next hidden state and reward
            predicted_next_hidden_states = self.dynamics_model(
                current_hidden_states, step_actions
            )
            predicted_rewards = self.reward_model(current_hidden_states, step_actions)

            # Get target next hidden states
            with torch.no_grad():
                target_next_hidden_states = self.policy.get_hidden_state(
                    step_next_states
                )

            # Compute losses
            dynamics_loss = nn.MSELoss()(
                predicted_next_hidden_states, target_next_hidden_states
            )
            reward_loss = self.reward_model.compute_loss(
                predicted_rewards, step_rewards
            )

            total_dynamics_loss += dynamics_loss
            total_reward_loss += reward_loss
            valid_steps += 1

            # Update hidden state for next step
            current_hidden_states = target_next_hidden_states

        # Average losses over sequence length
        avg_dynamics_loss = (
            total_dynamics_loss / valid_steps
            if valid_steps > 0
            else torch.tensor(0.0, device=states_seq.device, requires_grad=True)
        )
        avg_reward_loss = (
            total_reward_loss / valid_steps
            if valid_steps > 0
            else torch.tensor(0.0, device=states_seq.device, requires_grad=True)
        )

        return avg_dynamics_loss, avg_reward_loss

    def _get_regular_batch_iterator(self, exp_buffer):
        """Get regular batch iterator based on buffer capabilities."""
        if hasattr(exp_buffer, "get_all_batches_shuffled_muesli"):
            return exp_buffer.get_all_batches_shuffled_muesli(self.mini_batch_size)
        else:
            return exp_buffer.get_all_batches_shuffled(self.mini_batch_size)

    def _train_on_sequential_batch(self, sequential_batch):
        """
        Train on a sequential multi-step batch.

        Args:
            sequential_batch: Sequential batch data

        Returns:
            dict: Training metrics
        """
        # For now, implement a simplified version that processes sequential data
        # This would need to be expanded based on the specific sequential batch format

        # Extract the first timestep for policy learning (similar to regular batch)
        if isinstance(sequential_batch, dict):
            # Sequential batch format
            first_step_batch = (
                sequential_batch["actions"][:, 0],  # First actions
                sequential_batch["log_probs"][:, 0],  # First log_probs
                sequential_batch["states"][:, 0],  # First states
                sequential_batch["values"][:, 0],  # First values
                sequential_batch["advantages"][:, 0],  # First advantages
                sequential_batch["rewards"][:, 0],  # First rewards
                sequential_batch["next_states"][:, 0],  # First next_states
            )
        else:
            # Fallback to regular batch processing
            first_step_batch = sequential_batch

        # Process first step with regular method for policy learning
        batch_metrics = self._train_on_batch(first_step_batch)

        # Add sequential model learning if data supports it
        if isinstance(sequential_batch, dict):
            try:
                seq_model_loss, seq_reward_loss = self._compute_sequential_model_losses(
                    sequential_batch
                )
                batch_metrics["model_loss"] = (
                    float(seq_model_loss)
                    if isinstance(seq_model_loss, (int, float))
                    else seq_model_loss.item()
                )
                batch_metrics["reward_loss"] = (
                    float(seq_reward_loss)
                    if isinstance(seq_reward_loss, (int, float))
                    else seq_reward_loss.item()
                )
            except Exception as e:
                print(f"Warning: Sequential model learning failed: {e}")

        return batch_metrics

    def _compute_normalized_advantage(self, advantages, target_values):
        """
        Compute normalized advantages using exponential moving average variance.

        This follows the reference implementation approach for advantage normalization.

        Args:
            advantages (torch.Tensor): Raw advantages
            target_values (torch.Tensor): Target values for variance computation

        Returns:
            torch.Tensor: Normalized advantages
        """
        # Compute advantage variance using exponential moving average
        advantage_squared = (advantages**2).mean()

        # Update variance tracking
        self.advantage_var = (
            self.advantage_beta * self.advantage_var
            + (1 - self.advantage_beta) * advantage_squared
        )
        self.advantage_beta_product *= self.advantage_beta

        # Bias correction
        variance_hat = self.advantage_var / (1 - self.advantage_beta_product)

        # Normalize advantages
        normalized_advantages = advantages / (torch.sqrt(variance_hat) + 1e-8)

        return normalized_advantages

    def _compute_model_normalized_advantage(self, advantages, target_values, step_idx):
        """
        Compute normalized advantages for model learning at specific step.

        Args:
            advantages (torch.Tensor): Raw advantages for this step
            target_values (torch.Tensor): Target values
            step_idx (int): Step index for multi-step tracking

        Returns:
            torch.Tensor: Normalized advantages for this step
        """
        if step_idx >= len(self.model_advantage_vars):
            return advantages  # Fallback if step index is out of range

        # Compute variance for this specific step
        advantage_squared = (advantages**2).mean()

        # Update step-specific variance tracking
        self.model_advantage_vars[step_idx] = (
            self.advantage_beta * self.model_advantage_vars[step_idx]
            + (1 - self.advantage_beta) * advantage_squared
        )
        self.model_advantage_betas[step_idx] *= self.advantage_beta

        # Bias correction
        variance_hat = self.model_advantage_vars[step_idx] / (
            1 - self.model_advantage_betas[step_idx]
        )

        # Normalize advantages
        normalized_advantages = advantages / (torch.sqrt(variance_hat) + 1e-8)

        return normalized_advantages

    def save_to(self, folder_path):
        """Save all models to folder."""
        os.makedirs(folder_path, exist_ok=True)

        # Save main networks
        torch.save(
            self.policy.state_dict(), os.path.join(folder_path, "MUESLI_POLICY.pt")
        )
        torch.save(
            self.dynamics_model.state_dict(),
            os.path.join(folder_path, "MUESLI_DYNAMICS.pt"),
        )
        torch.save(
            self.reward_model.state_dict(),
            os.path.join(folder_path, "MUESLI_REWARD.pt"),
        )
        torch.save(
            self.prediction_head.state_dict(),
            os.path.join(folder_path, "MUESLI_PREDICTION.pt"),
        )

        # Save optimizers
        torch.save(
            self.policy_optimizer.state_dict(),
            os.path.join(folder_path, "MUESLI_POLICY_OPT.pt"),
        )
        torch.save(
            self.dynamics_optimizer.state_dict(),
            os.path.join(folder_path, "MUESLI_DYNAMICS_OPT.pt"),
        )
        torch.save(
            self.reward_optimizer.state_dict(),
            os.path.join(folder_path, "MUESLI_REWARD_OPT.pt"),
        )
        torch.save(
            self.prediction_optimizer.state_dict(),
            os.path.join(folder_path, "MUESLI_PREDICTION_OPT.pt"),
        )

        print(f"Muesli models saved to {folder_path}")

    def load_from(self, folder_path):
        """Load all models from folder."""
        assert os.path.exists(folder_path), f"Folder {folder_path} does not exist"

        # Load main networks
        self.policy.load_state_dict(
            torch.load(os.path.join(folder_path, "MUESLI_POLICY.pt"))
        )
        self.dynamics_model.load_state_dict(
            torch.load(os.path.join(folder_path, "MUESLI_DYNAMICS.pt"))
        )
        self.reward_model.load_state_dict(
            torch.load(os.path.join(folder_path, "MUESLI_REWARD.pt"))
        )
        self.prediction_head.load_state_dict(
            torch.load(os.path.join(folder_path, "MUESLI_PREDICTION.pt"))
        )

        # Load optimizers
        self.policy_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "MUESLI_POLICY_OPT.pt"))
        )
        self.dynamics_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "MUESLI_DYNAMICS_OPT.pt"))
        )
        self.reward_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "MUESLI_REWARD_OPT.pt"))
        )
        self.prediction_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "MUESLI_PREDICTION_OPT.pt"))
        )

        # Update target networks
        self.target_manager.hard_update_all_target_networks()

        print(f"Muesli models loaded from {folder_path}")
