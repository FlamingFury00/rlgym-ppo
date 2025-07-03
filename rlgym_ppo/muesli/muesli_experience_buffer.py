"""
Enhanced Experience Buffer for Muesli Algorithm

This buffer extends the basic experience buffer to support Muesli-specific features
like experience reanalysis and model-based data.
"""

import numpy as np
import torch
from collections import deque

from rlgym_ppo.ppo.experience_buffer import ExperienceBuffer


class MuesliExperienceBuffer(ExperienceBuffer):
    """
    Enhanced experience buffer for Muesli that supports additional data storage
    for model learning and experience reanalysis.
    """

    def __init__(
        self,
        max_size,
        seed,
        device,
        reanalyzer=None,
        sequence_length=5,
        enable_sequential=False,
    ):
        """
        Initialize the Muesli experience buffer.

        Args:
            max_size (int): Maximum buffer size
            seed (int): Random seed
            device (torch.device): Device for tensor operations
            reanalyzer (ExperienceReanalyzer): Optional experience reanalyzer
            sequence_length (int): Length of sequences to collect for multi-step learning
            enable_sequential (bool): Enable sequential data collection (can be memory intensive)
        """
        super().__init__(max_size, seed, device)
        self.reanalyzer = reanalyzer
        self.sequence_length = sequence_length
        self.enable_sequential = enable_sequential

        # Additional storage for Muesli-specific data
        self.hidden_states_buffer = deque(maxlen=max_size)
        self.trajectory_buffer = deque(maxlen=max_size // 10)  # Store trajectories

        # Sequential data storage for multi-step learning (only if enabled)
        if self.enable_sequential:
            # Much more conservative memory limits for sequential data
            max_sequences = min(1000, max_size // 50)  # Much smaller sequence storage

            self.current_episode_data = []  # Store current episode's transitions
            self.completed_sequences = deque(
                maxlen=max_sequences
            )  # Store completed sequences
            self.episode_step_count = 0
            self.max_episode_length = (
                500  # Limit episode length to prevent memory explosion
            )

            print(
                f"Sequential learning enabled: max {max_sequences} sequences, max episode length {self.max_episode_length}"
            )
        else:
            self.current_episode_data = None
            self.completed_sequences = None
            self.episode_step_count = 0
            self.max_episode_length = 0

    def submit_experience(
        self,
        states,
        actions,
        log_probs,
        rewards,
        next_states,
        dones,
        truncated,
        values,
        advantages,
        hidden_states=None,
    ):
        """
        Enhanced experience submission with hidden states and sequential data collection.

        Args:
            states: States from environment
            actions: Actions taken
            log_probs: Action log probabilities
            rewards: Rewards received
            next_states: Next states
            dones: Done flags
            truncated: Truncated flags
            values: Value estimates
            advantages: Advantage estimates
            hidden_states: Hidden state representations (for Muesli)
        """
        # Call parent implementation
        super().submit_experience(
            states,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            truncated,
            values,
            advantages,
        )

        # Store hidden states if provided
        if hidden_states is not None:
            np_hidden_states = np.asarray(hidden_states)
            for i in range(len(rewards)):
                self.hidden_states_buffer.append(np_hidden_states[i])
        else:
            # Add dummy hidden states to maintain alignment
            for i in range(len(rewards)):
                self.hidden_states_buffer.append(None)

        # Collect sequential data for multi-step learning (only if enabled)
        if self.enable_sequential:
            self._collect_sequential_data(
                states,
                actions,
                log_probs,
                rewards,
                next_states,
                dones,
                truncated,
                values,
                advantages,
                hidden_states,
            )

        # Store experience in reanalyzer if available
        if self.reanalyzer is not None:
            self.reanalyzer.store_experience(
                states, actions, rewards, next_states, dones, values, log_probs
            )

    def _collect_sequential_data(
        self,
        states,
        actions,
        log_probs,
        rewards,
        next_states,
        dones,
        truncated,
        values,
        advantages,
        hidden_states,
    ):
        """
        Collect sequential data for multi-step learning (memory-optimized version).

        Args:
            states, actions, log_probs, rewards, next_states, dones, truncated, values, advantages, hidden_states:
                Experience data from current step
        """
        # Early return if sequential data collection is disabled
        if not self.enable_sequential or self.current_episode_data is None:
            return

        # Prevent memory explosion by limiting episode length
        if len(self.current_episode_data) >= self.max_episode_length:
            print(
                f"Episode too long ({len(self.current_episode_data)} steps), finalizing sequences"
            )
            self._finalize_sequences()
            return

        # Convert to numpy arrays for consistent handling
        np_states = np.asarray(states)
        np_actions = np.asarray(actions)
        np_log_probs = np.asarray(log_probs)
        np_rewards = np.asarray(rewards)
        np_next_states = np.asarray(next_states)
        np_dones = np.asarray(dones)
        np_truncated = np.asarray(truncated)
        np_values = np.asarray(values)
        np_advantages = np.asarray(advantages)

        # Only process ONE agent at a time to reduce memory usage
        # For multi-agent environments, we'll just use the first agent
        batch_size = len(np_rewards)
        agent_idx = 0  # Only use first agent to save memory

        if batch_size > 0:
            # Create lightweight transition for first agent only
            transition = {
                "state": np_states[agent_idx] if np_states.ndim > 1 else np_states,
                "action": np_actions[agent_idx] if np_actions.ndim > 1 else np_actions,
                "log_prob": (
                    np_log_probs[agent_idx] if np_log_probs.ndim > 0 else np_log_probs
                ),
                "reward": np_rewards[agent_idx] if np_rewards.ndim > 0 else np_rewards,
                "next_state": (
                    np_next_states[agent_idx]
                    if np_next_states.ndim > 1
                    else np_next_states
                ),
                "done": np_dones[agent_idx] if np_dones.ndim > 0 else np_dones,
                "truncated": (
                    np_truncated[agent_idx] if np_truncated.ndim > 0 else np_truncated
                ),
                "value": np_values[agent_idx] if np_values.ndim > 0 else np_values,
                "advantage": (
                    np_advantages[agent_idx]
                    if np_advantages.ndim > 0
                    else np_advantages
                ),
                "hidden_state": (
                    hidden_states[agent_idx]
                    if hidden_states is not None and len(hidden_states) > agent_idx
                    else None
                ),
                "step": self.episode_step_count,
            }

            # Add to current episode
            self.current_episode_data.append(transition)

            # Check if episode is done or we have enough steps for a sequence
            done_flag = np_dones[agent_idx] if np_dones.ndim > 0 else np_dones
            truncated_flag = (
                np_truncated[agent_idx] if np_truncated.ndim > 0 else np_truncated
            )

            if (
                done_flag
                or truncated_flag
                or len(self.current_episode_data) >= self.sequence_length * 3
            ):
                self._finalize_sequences()

        self.episode_step_count += 1

    def _finalize_sequences(self):
        """
        Finalize sequences from current episode data (memory-optimized version).
        """
        if (
            not self.enable_sequential
            or self.current_episode_data is None
            or self.completed_sequences is None
        ):
            return

        if len(self.current_episode_data) < self.sequence_length:
            # Not enough data for a sequence, just clear
            self.current_episode_data.clear()
            self.episode_step_count = 0
            return

        # Create NON-OVERLAPPING sequences to save memory
        # Instead of creating overlapping sequences, create fewer, distinct sequences
        num_complete_sequences = len(self.current_episode_data) // self.sequence_length

        for seq_idx in range(
            min(num_complete_sequences, 3)
        ):  # Limit to max 3 sequences per episode
            start_idx = seq_idx * self.sequence_length
            end_idx = start_idx + self.sequence_length
            sequence = self.current_episode_data[start_idx:end_idx]

            # Convert sequence to batch format for efficient processing
            sequence_batch = {
                "states": np.array([t["state"] for t in sequence]),
                "actions": np.array([t["action"] for t in sequence]),
                "log_probs": np.array([t["log_prob"] for t in sequence]),
                "rewards": np.array([t["reward"] for t in sequence]),
                "next_states": np.array([t["next_state"] for t in sequence]),
                "dones": np.array([t["done"] for t in sequence]),
                "truncated": np.array([t["truncated"] for t in sequence]),
                "values": np.array([t["value"] for t in sequence]),
                "advantages": np.array([t["advantage"] for t in sequence]),
                "hidden_states": np.array(
                    [
                        t["hidden_state"]
                        for t in sequence
                        if t["hidden_state"] is not None
                    ]
                ),
                "sequence_length": self.sequence_length,
            }

            self.completed_sequences.append(sequence_batch)

        # Clear episode data and reset counter
        self.current_episode_data.clear()
        self.episode_step_count = 0

        # Print memory usage info
        if len(self.completed_sequences) % 100 == 0:
            print(
                f"Sequential buffer: {len(self.completed_sequences)} sequences stored"
            )

    def _get_muesli_samples(self, indices):
        """
        Get samples with additional Muesli data (next_states and rewards).

        Args:
            indices: Indices of samples to extract

        Returns:
            tuple: (actions, log_probs, states, values, advantages, rewards, next_states)
        """
        batch = [self.buffer[i] for i in indices]

        actions = torch.as_tensor(
            np.array([item[1] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        log_probs = torch.as_tensor(
            np.array([item[2] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        states = torch.as_tensor(
            np.array([item[0] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        values = torch.as_tensor(
            np.array([item[7] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.as_tensor(
            np.array([item[8] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        # Add rewards and next_states for Muesli model learning
        rewards = torch.as_tensor(
            np.array([item[3] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            np.array([item[4] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        return actions, log_probs, states, values, advantages, rewards, next_states

    def get_all_batches_shuffled_muesli(self, batch_size):
        """
        Get Muesli batches with additional data (rewards and next_states).

        Args:
            batch_size: Size of each batch

        Yields:
            tuple: Muesli batch with rewards and next_states
        """
        total_samples = len(self.buffer)
        if total_samples == 0:
            return

        indices = np.arange(total_samples)
        self.rng.shuffle(indices)

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield self._get_muesli_samples(batch_indices)

    def get_sequential_batches(self, batch_size, sequence_length=None):
        """
        Get sequential batches for multi-step learning.

        Args:
            batch_size: Number of sequences per batch
            sequence_length: Length of sequences (uses default if None)

        Yields:
            dict: Sequential batch data
        """
        if not self.enable_sequential or self.completed_sequences is None:
            return

        if sequence_length is None:
            sequence_length = self.sequence_length

        if len(self.completed_sequences) == 0:
            return

        # Convert completed sequences to list for indexing
        sequences = list(self.completed_sequences)
        total_sequences = len(sequences)

        # Shuffle sequences
        indices = np.arange(total_sequences)
        self.rng.shuffle(indices)

        for start_idx in range(0, total_sequences, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            if len(batch_indices) == 0:
                break

            # Collect sequences for this batch
            batch_sequences = [sequences[i] for i in batch_indices]

            # Stack sequences into batch format
            batch_data = self._stack_sequences(batch_sequences)
            yield batch_data

    def _stack_sequences(self, sequences):
        """
        Stack multiple sequences into a single batch.

        Args:
            sequences: List of sequence dictionaries

        Returns:
            dict: Stacked batch data
        """
        if not sequences:
            return {}

        # Initialize batch tensors
        batch_data = {}
        keys_to_stack = [
            "states",
            "actions",
            "log_probs",
            "rewards",
            "next_states",
            "values",
            "advantages",
            "dones",
            "truncated",
        ]

        # Stack each field
        for key in keys_to_stack:
            field_data = []
            for seq in sequences:
                if key in seq:
                    field_data.append(seq[key])

            if field_data:
                stacked = np.stack(
                    field_data, axis=0
                )  # Shape: (batch_size, seq_len, ...)
                batch_data[key] = torch.as_tensor(
                    stacked, dtype=torch.float32, device=self.device
                )

        return batch_data

    def get_reanalyzed_batch(
        self, batch_size, dynamics_model, reward_model, value_model, policy_model
    ):
        """
        Get a batch with reanalyzed experiences mixed in.

        Args:
            batch_size (int): Size of batch to generate
            dynamics_model: Current dynamics model
            reward_model: Current reward model
            value_model: Current value model
            policy_model: Current policy model

        Returns:
            generator: Batches with mixed fresh and reanalyzed experiences
        """
        if self.reanalyzer is None:
            # Fallback to normal batches
            yield from self.get_all_batches_shuffled(batch_size)
            return

        # Get reanalysis ratio
        reanalysis_ratio = self.reanalyzer.reanalysis_ratio
        reanalyzed_size = int(batch_size * reanalysis_ratio)
        fresh_size = batch_size - reanalyzed_size

        total_samples = len(self.buffer)
        if total_samples == 0:
            return

        indices = np.arange(total_samples)
        self.rng.shuffle(indices)

        for start_idx in range(0, total_samples, fresh_size):
            # Get fresh experiences
            fresh_indices = indices[start_idx : start_idx + fresh_size]
            if len(fresh_indices) == 0:
                break

            fresh_batch = self._get_samples(fresh_indices)

            # Get reanalyzed experiences if available
            if reanalyzed_size > 0 and len(self.reanalyzer.replay_buffer) > 0:
                reanalyzed_experiences = (
                    self.reanalyzer.sample_experiences_for_reanalysis(reanalyzed_size)
                )

                if reanalyzed_experiences:
                    reanalyzed_data = self.reanalyzer.reanalyze_experiences(
                        reanalyzed_experiences,
                        dynamics_model,
                        reward_model,
                        value_model,
                        policy_model,
                    )

                    if reanalyzed_data:
                        # Combine fresh and reanalyzed data
                        combined_batch = self._combine_batches(
                            fresh_batch, reanalyzed_data
                        )
                        yield combined_batch
                        continue

            # If no reanalyzed data available, yield fresh batch
            yield fresh_batch

    def _combine_batches(self, fresh_batch, reanalyzed_data):
        """
        Combine fresh and reanalyzed batches.

        Args:
            fresh_batch: Fresh experience batch
            reanalyzed_data: Reanalyzed experience data

        Returns:
            tuple: Combined batch
        """
        actions, log_probs, states, values, advantages = fresh_batch

        # Extract reanalyzed data
        r_states = reanalyzed_data["states"]
        r_actions = reanalyzed_data["actions"]
        r_log_probs = reanalyzed_data["log_probs"]
        r_values = reanalyzed_data["values"]
        r_advantages = reanalyzed_data["advantages"]

        # Concatenate tensors
        combined_actions = torch.cat([actions, r_actions], dim=0)
        combined_log_probs = torch.cat([log_probs, r_log_probs], dim=0)
        combined_states = torch.cat([states, r_states], dim=0)
        combined_values = torch.cat([values, r_values], dim=0)
        combined_advantages = torch.cat([advantages, r_advantages], dim=0)

        return (
            combined_actions,
            combined_log_probs,
            combined_states,
            combined_values,
            combined_advantages,
        )

    def get_trajectory_data(self, trajectory_length=5):
        """
        Get sequential trajectory data for model learning.

        Args:
            trajectory_length (int): Length of trajectories to extract

        Returns:
            list: List of trajectory dictionaries
        """
        trajectories = []

        if len(self.buffer) < trajectory_length:
            return trajectories

        # Extract sequential trajectories
        for i in range(len(self.buffer) - trajectory_length + 1):
            trajectory = []

            for j in range(trajectory_length):
                experience = self.buffer[i + j]
                trajectory.append(
                    {
                        "state": experience[0],
                        "action": experience[1],
                        "log_prob": experience[2],
                        "reward": experience[3],
                        "next_state": experience[4],
                        "done": experience[5],
                        "truncated": experience[6],
                        "value": experience[7],
                        "advantage": experience[8],
                    }
                )

            trajectories.append(trajectory)

        return trajectories

    def clear(self):
        """Clear all buffers including sequential data."""
        super().clear()
        self.hidden_states_buffer.clear()
        self.trajectory_buffer.clear()
        if self.enable_sequential and self.current_episode_data is not None:
            self.current_episode_data.clear()
        if self.enable_sequential and self.completed_sequences is not None:
            self.completed_sequences.clear()
        self.episode_step_count = 0
