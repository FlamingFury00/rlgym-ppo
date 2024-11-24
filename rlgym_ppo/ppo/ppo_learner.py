# ppo_learner.py
import os
import time

import numpy as np
import torch

from rlgym_ppo.ppo.shared_network import SharedNetwork


class PPOLearner(object):
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        policy_type,
        layer_sizes,
        continuous_var_range,
        batch_size,
        n_epochs,
        learning_rate,
        clip_range,
        ent_coef,
        mini_batch_size,
        device,
    ):
        self.device = device

        assert (
            batch_size % mini_batch_size == 0
        ), "MINIBATCH SIZE MUST BE AN INTEGER MULTIPLE OF BATCH SIZE"

        # Instantiate the shared network
        # For discrete action spaces only in this case
        self.shared_network = SharedNetwork(
            obs_space_size,
            act_space_size,
            layer_sizes,
            device,
        ).to(device)

        # Single optimizer for the shared network
        self.optimizer = torch.optim.Adam(
            self.shared_network.parameters(), lr=learning_rate
        )
        self.value_loss_fn = torch.nn.MSELoss()

        # Calculate parameter counts
        params = self.shared_network.parameters()
        trainable_parameters = filter(lambda p: p.requires_grad, params)
        total_parameters = sum(p.numel() for p in trainable_parameters)

        print("Trainable Parameters: {}".format(total_parameters))
        print(f"Current Learning Rate: {learning_rate}")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.mini_batch_size = mini_batch_size
        self.cumulative_model_updates = 0

    def learn(self, exp):
        """
        Compute PPO updates with an experience buffer.

        Args:
            exp (ExperienceBuffer): Experience buffer containing training data.

        Returns:
            dict: Dictionary containing training report metrics.
        """
        n_iterations = 0
        n_minibatch_iterations = 0
        mean_entropy = 0
        mean_divergence = 0
        mean_val_loss = 0
        clip_fractions = []

        # Save parameters before computing any updates.
        params_before = torch.nn.utils.parameters_to_vector(
            self.shared_network.parameters()
        ).cpu()

        t1 = time.time()
        for epoch in range(self.n_epochs):
            # Get all shuffled batches from the experience buffer.
            batches = exp.get_all_batches_shuffled(self.batch_size)
            for batch in batches:
                (
                    batch_acts,
                    batch_old_probs,
                    batch_obs,
                    batch_target_values,
                    batch_advantages,
                ) = batch
                batch_acts = batch_acts.view(self.batch_size, -1)

                for minibatch_slice in range(0, self.batch_size, self.mini_batch_size):
                    # Send everything to the device and enforce correct shapes.
                    start = minibatch_slice
                    stop = start + self.mini_batch_size

                    acts = batch_acts[start:stop].to(self.device)
                    obs = batch_obs[start:stop].to(self.device)
                    advantages = batch_advantages[start:stop].to(self.device)
                    old_probs = batch_old_probs[start:stop].to(self.device)
                    target_values = batch_target_values[start:stop].to(self.device)

                    # Compute policy log probs, entropy, and value estimates.
                    log_probs, entropy = self.shared_network.get_backprop_data(
                        obs, acts
                    )
                    log_probs = log_probs.view_as(old_probs)
                    _, vals = self.shared_network.get_output(obs)
                    vals = vals.view_as(target_values)

                    # Compute PPO loss.
                    ratio = torch.exp(log_probs - old_probs)
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                    )

                    # Compute KL divergence & clip fraction using SB3 method for reporting.
                    with torch.no_grad():
                        log_ratio = log_probs - old_probs
                        kl = (torch.exp(log_ratio) - 1) - log_ratio
                        kl = kl.mean().detach().cpu().item()

                        # From the stable-baselines3 implementation of PPO.
                        clip_fraction = (
                            torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
                            .cpu()
                            .item()
                        )
                        clip_fractions.append(clip_fraction)

                    policy_loss = -torch.min(
                        ratio * advantages, clipped * advantages
                    ).mean()
                    value_loss = self.value_loss_fn(vals, target_values)

                    ppo_loss = policy_loss - entropy * self.ent_coef + value_loss

                    ppo_loss.backward()

                    mean_val_loss += value_loss.cpu().detach().item()
                    mean_divergence += kl
                    mean_entropy += entropy.cpu().detach().item()
                    n_minibatch_iterations += 1

                torch.nn.utils.clip_grad_norm_(
                    self.shared_network.parameters(), max_norm=0.5
                )

                self.optimizer.step()
                self.optimizer.zero_grad()

                n_iterations += 1

        if n_iterations == 0:
            n_iterations = 1

        if n_minibatch_iterations == 0:
            n_minibatch_iterations = 1

        # Compute averages for the metrics that will be reported.
        mean_entropy /= n_minibatch_iterations
        mean_divergence /= n_minibatch_iterations
        mean_val_loss /= n_minibatch_iterations
        if len(clip_fractions) == 0:
            mean_clip = 0
        else:
            mean_clip = np.mean(clip_fractions)

        # Compute magnitude of updates made to the shared network.
        params_after = torch.nn.utils.parameters_to_vector(
            self.shared_network.parameters()
        ).cpu()
        update_magnitude = (params_before - params_after).norm().item()

        # Assemble and return report dictionary.
        self.cumulative_model_updates += n_iterations

        report = {
            "PPO Batch Consumption Time": (time.time() - t1) / n_iterations,
            "Cumulative Model Updates": self.cumulative_model_updates,
            "Policy Entropy": mean_entropy,
            "Mean KL Divergence": mean_divergence,
            "Value Function Loss": mean_val_loss,
            "SB3 Clip Fraction": mean_clip,
            "Update Magnitude": update_magnitude,
        }

        return report

    def save_to(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(
            self.shared_network.state_dict(),
            os.path.join(folder_path, "PPO_SHARED_NETWORK.pt"),
        )
        torch.save(
            self.optimizer.state_dict(), os.path.join(folder_path, "PPO_OPTIMIZER.pt")
        )

    def load_from(self, folder_path):
        assert os.path.exists(folder_path), "PPO LEARNER CANNOT FIND FOLDER {}".format(
            folder_path
        )

        self.shared_network.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_SHARED_NETWORK.pt"))
        )
        self.optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_OPTIMIZER.pt"))
        )
