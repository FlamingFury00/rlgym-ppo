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
        sequence_length=5,
        enable_sequential=False,
        replay_buffer_size=100000,  # Default size, can be configured
        reanalysis_ratio=0.0,  # Default to no reanalysis
    ):
        """
        Initialize the Muesli experience buffer.

        Args:
            max_size (int): Maximum buffer size for fresh experiences
            seed (int): Random seed
            device (torch.device): Device for tensor operations
            sequence_length (int): Length of sequences to collect for multi-step learning
            enable_sequential (bool): Enable sequential data collection
            replay_buffer_size (int): Maximum size of the replay buffer for reanalysis
            reanalysis_ratio (float): Ratio of reanalyzed to fresh experiences in batches
        """
        super().__init__(max_size, seed, device)
        self.sequence_length = sequence_length
        self.enable_sequential = enable_sequential

        # Replay buffer for experience reanalysis
        self.replay_buffer_size = replay_buffer_size
        self.reanalysis_ratio = reanalysis_ratio
        if self.reanalysis_ratio > 0:
            self.replay_buffer = deque(maxlen=self.replay_buffer_size)
            self.total_experiences_stored_for_reanalysis = 0
            self.total_reanalysis_samples_provided = 0
        else:
            self.replay_buffer = None  # No replay buffer if ratio is zero

        # Additional storage for Muesli-specific data
        self.hidden_states_buffer = deque(maxlen=max_size)

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

        # Store experience in the replay buffer for reanalysis, if active
        if self.replay_buffer is not None:
            # Iterate over each transition in the batch
            # Assuming states, actions etc. are lists/arrays of individual agent experiences
            # or a batch tensor which needs to be iterated.
            # For simplicity, let's assume they are already batched and we process each item.

            num_transitions = len(
                rewards
            )  # states, actions etc. should have same first dimension

            # Convert tensors to cpu and then to numpy for storage, if they are tensors
            # This helps in reducing GPU memory if buffer grows large and if tensors were on GPU
            np_states = (
                states.cpu().numpy()
                if isinstance(states, torch.Tensor)
                else np.asarray(states)
            )
            np_actions = (
                actions.cpu().numpy()
                if isinstance(actions, torch.Tensor)
                else np.asarray(actions)
            )
            np_log_probs = (
                log_probs.cpu().numpy()
                if isinstance(log_probs, torch.Tensor)
                else np.asarray(log_probs)
            )
            np_rewards = (
                rewards.cpu().numpy()
                if isinstance(rewards, torch.Tensor)
                else np.asarray(rewards)
            )
            np_next_states = (
                next_states.cpu().numpy()
                if isinstance(next_states, torch.Tensor)
                else np.asarray(next_states)
            )
            np_dones = (
                dones.cpu().numpy()
                if isinstance(dones, torch.Tensor)
                else np.asarray(dones)
            )
            np_values = (
                values.cpu().numpy()
                if isinstance(values, torch.Tensor)
                else np.asarray(values)
            )
            # hidden_states are optional and might be complex; store as is or a serializable form
            # For now, let's assume hidden_states are numpy arrays or None

            for i in range(num_transitions):
                experience = {
                    "state": np_states[i],
                    "action": np_actions[i],
                    "log_prob": np_log_probs[i],
                    "reward": np_rewards[i],
                    "next_state": np_next_states[i],
                    "done": np_dones[i],
                    "value": np_values[
                        i
                    ],  # Original value estimate at time of experience
                }
                self.replay_buffer.append(experience)
                self.total_experiences_stored_for_reanalysis += 1

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

    def _sample_for_reanalysis(self, num_samples: int) -> list[dict]:
        """
        Sample experiences from the replay buffer for reanalysis.

        Args:
            num_samples (int): Number of experiences to sample.

        Returns:
            list[dict]: A list of sampled experience dictionaries.
                        Returns an empty list if buffer is empty or not active.
        """
        if not self.replay_buffer or len(self.replay_buffer) == 0:
            return []

        actual_num_samples = min(num_samples, len(self.replay_buffer))

        # Efficiently sample random indices using numpy's random generator unique to this buffer
        sampled_indices = self.rng.choice(
            len(self.replay_buffer), size=actual_num_samples, replace=False
        )

        sampled_experiences = [self.replay_buffer[i] for i in sampled_indices]

        if hasattr(self, "total_reanalysis_samples_provided"):
            self.total_reanalysis_samples_provided += actual_num_samples
        return sampled_experiences

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

    def clear(self):
        """Clear all buffers including sequential data."""
        super().clear()
        self.hidden_states_buffer.clear()
        if self.enable_sequential and self.current_episode_data is not None:
            self.current_episode_data.clear()
        if self.enable_sequential and self.completed_sequences is not None:
            self.completed_sequences.clear()
        self.episode_step_count = 0
