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
        Note: This method now uses `np.float16` for memory efficiency and assigns a priority to each experience based on its advantage.
        """
        np_states = np.asarray(states, dtype=np.float16)
        np_actions = np.asarray(actions, dtype=np.float16)
        np_log_probs = np.asarray(log_probs, dtype=np.float16)
        np_rewards = np.asarray(rewards, dtype=np.float16)
        np_next_states = np.asarray(next_states, dtype=np.float16)
        np_dones = np.asarray(dones, dtype=np.float16)
        np_truncated = np.asarray(truncated, dtype=np.float16)
        np_values = np.asarray(values, dtype=np.float16)
        np_advantages = np.asarray(advantages, dtype=np.float16)

        for i in range(len(np_rewards)):
            priority = abs(np_advantages[i]) + 1e-6  # Priority based on advantage magnitude
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
                    priority,
                )
            )

    def _get_samples(self, indices):
        # Normalize priorities to sum to 1
        priorities = np.array([item[-1] for item in self.buffer], dtype=np.float32)
        probabilities = priorities / priorities.sum()

        # Sample indices based on probabilities
        sampled_indices = self.rng.choice(len(self.buffer), size=len(indices), p=probabilities)
        batch = [self.buffer[i] for i in sampled_indices]

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
        :return: Generator yielding batches of experiences. Experiences are sampled with probabilities proportional to their priorities.
        """
        total_samples = len(self.buffer)
        if total_samples == 0:
            return

        # Use prioritized sampling for indices
        priorities = np.array([item[-1] for item in self.buffer], dtype=np.float32)
        probabilities = priorities / priorities.sum()
        indices = self.rng.choice(total_samples, size=total_samples, replace=False, p=probabilities)

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_indices)

    def clear(self):
        """
        Function to clear the experience buffer.
        :return: None.
        """
        self.buffer.clear