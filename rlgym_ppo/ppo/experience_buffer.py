import numpy as np
import torch


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Update the sum tree iteratively from a leaf node to the root."""
        while idx != 0:  # Continue until we reach the root
            parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class ExperienceBuffer(object):
    def __init__(
        self, max_size, seed, device, alpha=0.6, beta=0.4, beta_increment=0.001
    ):
        self.device = device
        self.seed = seed
        self.max_size = max_size
        self.tree = SumTree(max_size)
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta  # Importance sampling correction
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.rng = np.random.RandomState(seed)

        # Original tensors
        self.states = torch.FloatTensor().to(self.device)
        self.actions = torch.FloatTensor().to(self.device)
        self.log_probs = torch.FloatTensor().to(self.device)
        self.rewards = torch.FloatTensor().to(self.device)
        self.next_states = torch.FloatTensor().to(self.device)
        self.dones = torch.FloatTensor().to(self.device)
        self.truncated = torch.FloatTensor().to(self.device)
        self.values = torch.FloatTensor().to(self.device)
        self.advantages = torch.FloatTensor().to(self.device)

    @staticmethod
    def _cat(t1, t2, size):
        if len(t2) > size:
            t = t2[-size:].clone()
        elif len(t2) == size:
            t = t2
        elif len(t1) + len(t2) > size:
            t = torch.cat((t1[len(t2) - size :], t2), 0)
        else:
            t = torch.cat((t1, t2), 0)
        del t1
        del t2
        return t

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
        _cat = ExperienceBuffer._cat

        # Store experience with max priority for new samples
        for i in range(len(states)):
            experience = (
                states[i],
                actions[i],
                log_probs[i],
                rewards[i],
                next_states[i],
                dones[i],
                truncated[i],
                values[i],
                advantages[i],
            )
            self.tree.add(self.max_priority, experience)

        # Update tensors as before
        self.states = _cat(
            self.states,
            torch.as_tensor(states, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.actions = _cat(
            self.actions,
            torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.log_probs = _cat(
            self.log_probs,
            torch.as_tensor(log_probs, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.rewards = _cat(
            self.rewards,
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.next_states = _cat(
            self.next_states,
            torch.as_tensor(next_states, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.dones = _cat(
            self.dones,
            torch.as_tensor(dones, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.truncated = _cat(
            self.truncated,
            torch.as_tensor(truncated, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.values = _cat(
            self.values,
            torch.as_tensor(values, dtype=torch.float32, device=self.device),
            self.max_size,
        )
        self.advantages = _cat(
            self.advantages,
            torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            self.max_size,
        )

    def _get_priority(self, error):
        return (np.abs(error) + 1e-5) ** self.alpha

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def _get_samples(self, indices):
        return (
            self.actions[indices],
            self.log_probs[indices],
            self.states[indices],
            self.values[indices],
            self.advantages[indices],
        )

    def get_all_batches_shuffled(self, batch_size):
        self.beta = min(1.0, self.beta + self.beta_increment)

        total_samples = self.rewards.shape[0]
        indices = self.rng.permutation(total_samples)
        start_idx = 0

        while start_idx + batch_size <= total_samples:
            batch_indices = indices[start_idx : start_idx + batch_size]

            # Calculate importance sampling weights
            priorities = self.tree.tree[batch_indices + self.tree.capacity - 1]
            probabilities = priorities / (
                self.tree.total() + 1e-5
            )  # Add small epsilon to avoid division by zero

            # Add small epsilon to probabilities to avoid zero values
            probabilities = np.maximum(probabilities, 1e-5)

            weights = (probabilities * total_samples) ** (-self.beta)
            weights = weights / (
                weights.max() + 1e-5
            )  # Add small epsilon to avoid division by zero

            yield self._get_samples(batch_indices), weights, batch_indices
            start_idx += batch_size

    def clear(self):
        del self.states
        del self.actions
        del self.log_probs
        del self.rewards
        del self.next_states
        del self.dones
        del self.truncated
        del self.values
        del self.advantages
        self.__init__(self.max_size, self.seed, self.device)
