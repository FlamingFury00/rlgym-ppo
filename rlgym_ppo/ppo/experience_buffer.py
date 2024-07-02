import numpy as np
import torch

from rlgym_ppo.ppo.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedExperienceBuffer(object):
    @staticmethod
    def _cat(t1, t2, size):
        if len(t2) > size:
            # t2 alone is larger than we want; copy the end
            t = t2[-size:].clone()
        elif len(t2) == size:
            # t2 is a perfect match; just use it directly
            t = t2
        elif len(t1) + len(t2) > size:
            # t1+t2 is larger than we want; use t2 wholly with the end of t1 before it
            t = torch.cat((t1[len(t2) - size :], t2), 0)
        else:
            # t1+t2 does not exceed what we want; concatenate directly
            t = torch.cat((t1, t2), 0)

        del t1
        del t2
        return t

    def __init__(self, max_size, seed, device, alpha=0.6):
        self.device = device
        self.seed = seed
        self.max_size = max_size
        self.alpha = alpha  # How much prioritization is used (0 - no prioritization, 1 - full prioritization)

        self.states = torch.FloatTensor().to(self.device)
        self.actions = torch.FloatTensor().to(self.device)
        self.log_probs = torch.FloatTensor().to(self.device)
        self.rewards = torch.FloatTensor().to(self.device)
        self.next_states = torch.FloatTensor().to(self.device)
        self.dones = torch.FloatTensor().to(self.device)
        self.truncated = torch.FloatTensor().to(self.device)
        self.values = torch.FloatTensor().to(self.device)
        self.advantages = torch.FloatTensor().to(self.device)

        it_capacity = 1
        while it_capacity < max_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
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
        _cat = PrioritizedExperienceBuffer._cat
        new_size = len(states)

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

        # Add priorities for new experiences
        current_size = len(self.states)
        for i in range(new_size):
            idx = (current_size - new_size + i) % self.max_size
            self._it_sum[idx] = self._max_priority**self.alpha
            self._it_min[idx] = self._max_priority**self.alpha

    def _get_samples(self, indices):
        return (
            self.actions[indices],
            self.log_probs[indices],
            self.states[indices],
            self.values[indices],
            self.advantages[indices],
        )

    def sample(self, batch_size, beta=0.4):
        indices = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.states)) ** (-beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.states)) ** (-beta)
            weights.append(weight / max_weight)

        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        samples = self._get_samples(indices)
        return tuple(list(samples) + [weights, indices])

    def _sample_proportional(self, batch_size):
        indices = []
        p_total = self._it_sum.sum(0, len(self.states) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = self.rng.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            indices.append(idx)
        return indices

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.states)
            self._it_sum[idx] = priority**self.alpha
            self._it_min[idx] = priority**self.alpha
            self._max_priority = max(self._max_priority, priority)

    def clear(self):
        self.states = torch.FloatTensor().to(self.device)
        self.actions = torch.FloatTensor().to(self.device)
        self.log_probs = torch.FloatTensor().to(self.device)
        self.rewards = torch.FloatTensor().to(self.device)
        self.next_states = torch.FloatTensor().to(self.device)
        self.dones = torch.FloatTensor().to(self.device)
        self.truncated = torch.FloatTensor().to(self.device)
        self.values = torch.FloatTensor().to(self.device)
        self.advantages = torch.FloatTensor().to(self.device)

        it_capacity = 1
        while it_capacity < self.max_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def get_all_batches_shuffled(self, batch_size):
        total_samples = len(self.rewards)
        indices = self.rng.permutation(total_samples)
        start_idx = 0
        while start_idx + batch_size <= total_samples:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
