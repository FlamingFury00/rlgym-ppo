"""
File: experience_buffer.py
Author: Matthew Allen

Description:
    A buffer containing the experience to be learned from on this iteration. The buffer may be added to, removed from,
    and shuffled. When the maximum specified size of the buffer is exceeded, the least recent entries will be removed in
    a FIFO fashion.
"""

from collections import deque

import numpy as np
import torch


class ExperienceBuffer(object):
    def __init__(self, max_size, seed, device):
        self.device = device
        self.seed = seed
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)  # Use deque for efficient FIFO
        self.rng = np.random.RandomState(seed)

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
    ):
        """
        Function to add experience to the buffer.

        :param states: An ordered sequence of states from the environment.
        :param actions: The corresponding actions that were taken at each state in the `states` sequence.
        :param log_probs: The log probability for each action in `actions`
        :param rewards: A list rewards for each pair in `states` and `actions`
        :param next_states: An ordered sequence of next states (the states which occurred after an action) from the environment.
        :param dones: An ordered sequence of the done (terminated) flag from the environment.
        :param truncated: An ordered sequence of the truncated flag from the environment.
        :param values: The output of the value function estimator evaluated on the concatenation of `states` and the final state in `next_states`
        :param advantages: The advantage of each action at each state in `states` and `actions`

        :return: None
        """
        np_states = np.asarray(states)
        np_actions = np.asarray(actions)
        np_log_probs = np.asarray(log_probs)
        np_rewards = np.asarray(rewards)
        np_next_states = np.asarray(next_states)
        np_dones = np.asarray(dones)
        np_truncated = np.asarray(truncated)
        np_values = np.asarray(values)
        np_advantages = np.asarray(advantages)

        for i in range(len(np_rewards)):
            self.buffer.append(
                (
                    np_states[i],
                    np_actions[i],
                    np_log_probs[i],
                    np_rewards[i],
                    np_next_states[i],
                    np_dones[i],
                    np_truncated[i],
                    np_values[i],
                    np_advantages[i],
                )
            )

    def _get_samples(self, indices):
        batch = [self.buffer[i] for i in indices]
        # Stack the numpy arrays and convert to tensors once
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
        return actions, log_probs, states, values, advantages

    def get_all_batches_shuffled(self, batch_size):
        """
        Function to return the experience buffer in shuffled batches.
        :param batch_size: size of each batch yielded by the generator.
        :return:
        """
        total_samples = len(self.buffer)
        if total_samples == 0:
            return

        indices = np.arange(total_samples)
        self.rng.shuffle(indices)

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_indices)

    def clear(self):
        """
        Function to clear the experience buffer.
        :return: None.
        """
        self.buffer.clear()
