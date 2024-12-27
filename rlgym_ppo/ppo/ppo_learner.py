import os
import time

import numpy as np
import torch

from rlgym_ppo.ppo import ContinuousPolicy, DiscreteFF, MultiDiscreteFF, ValueEstimator


class PPOLearner(object):
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        policy_type,
        policy_layer_sizes,
        critic_layer_sizes,
        continuous_var_range,
        batch_size,
        n_epochs,
        policy_lr,
        critic_lr,
        clip_range,
        ent_coef,
        mini_batch_size,
        device,
    ):
        self.device = device

        assert (
            batch_size % mini_batch_size == 0
        ), "MINIBATCH SIZE MUST BE AN INTEGER MULTIPLE OF BATCH SIZE"

        if policy_type == 2:
            self.policy = ContinuousPolicy(
                obs_space_size,
                act_space_size * 2,
                policy_layer_sizes,
                device,
                var_min=continuous_var_range[0],
                var_max=continuous_var_range[1],
            ).to(device)
        elif policy_type == 1:
            self.policy = MultiDiscreteFF(
                obs_space_size, policy_layer_sizes, device
            ).to(device)
        else:
            self.policy = DiscreteFF(
                obs_space_size, act_space_size, policy_layer_sizes, device
            ).to(device)
        self.value_net = ValueEstimator(obs_space_size, critic_layer_sizes, device).to(
            device
        )
        self.mini_batch_size = mini_batch_size
        self.num_mini_batches = batch_size // mini_batch_size  # Pre-calculate

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=critic_lr
        )
        self.value_loss_fn = torch.nn.MSELoss()

        # Calculate and print parameter counts (moved to init for clarity)
        policy_params_count = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        critic_params_count = sum(
            p.numel() for p in self.value_net.parameters() if p.requires_grad
        )
        total_parameters = policy_params_count + critic_params_count

        print("Trainable Parameters:")
        print(f"{'Component':<10} {'Count':<10}")
        print("-" * 20)
        print(f"{'Policy':<10} {policy_params_count:<10}")
        print(f"{'Critic':<10} {critic_params_count:<10}")
        print("-" * 20)
        print(f"{'Total':<10} {total_parameters:<10}")

        print(f"Current Policy Learning Rate: {policy_lr}")
        print(f"Current Critic Learning Rate: {critic_lr}")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
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
        mean_entropy = 0.0
        mean_divergence = 0.0
        mean_val_loss = 0.0
        clip_fractions = []

        policy_before = torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        ).cpu()
        critic_before = torch.nn.utils.parameters_to_vector(
            self.value_net.parameters()
        ).cpu()

        t1 = time.time()
        for epoch in range(self.n_epochs):
            for batch in exp.get_all_batches_shuffled(
                self.mini_batch_size
            ):  # Iterate over minibatches directly
                (
                    batch_acts,
                    batch_old_probs,
                    batch_obs,
                    batch_target_values,
                    batch_advantages,
                ) = batch
                batch_obs = batch_obs.to(self.device)
                batch_acts = batch_acts.view(batch_acts.size(0), -1).to(self.device)
                batch_old_probs = batch_old_probs.to(self.device)
                batch_target_values = batch_target_values.to(self.device)
                batch_advantages = batch_advantages.to(self.device)

                self.policy_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)

                # Compute value estimates.
                vals = self.value_net(batch_obs).view_as(batch_target_values)

                # Get policy log probs & entropy.
                log_probs, entropy = self.policy.get_backprop_data(
                    batch_obs, batch_acts
                )
                log_probs = log_probs.view_as(batch_old_probs)

                # Compute PPO loss.
                ratio = torch.exp(log_probs - batch_old_probs)
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                policy_loss = -torch.min(
                    ratio * batch_advantages, clipped * batch_advantages
                ).mean()

                value_loss = self.value_loss_fn(vals, batch_target_values)
                ppo_loss = policy_loss - entropy * self.ent_coef

                ppo_loss.backward()
                value_loss.backward()

                mean_val_loss += value_loss.item()
                mean_entropy += entropy.item()

                # Compute KL divergence & clip fraction using SB3 method for reporting.
                with torch.no_grad():
                    log_ratio = log_probs - batch_old_probs
                    kl = (torch.exp(log_ratio) - 1) - log_ratio
                    mean_divergence += kl.mean().cpu().item()
                    clip_fraction = (
                        torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
                        .cpu()
                        .item()
                    )
                    clip_fractions.append(clip_fraction)

                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), max_norm=0.5
                )
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

                self.policy_optimizer.step()
                self.value_optimizer.step()
                n_iterations += 1

        if n_iterations == 0:
            return {}  # Avoid division by zero

        mean_entropy /= n_iterations
        mean_divergence /= n_iterations
        mean_val_loss /= n_iterations
        mean_clip = np.mean(clip_fractions) if clip_fractions else 0

        policy_after = torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        ).cpu()
        critic_after = torch.nn.utils.parameters_to_vector(
            self.value_net.parameters()
        ).cpu()
        policy_update_magnitude = (policy_before - policy_after).norm().item()
        critic_update_magnitude = (critic_before - critic_after).norm().item()

        self.cumulative_model_updates += n_iterations

        report = {
            "PPO Batch Consumption Time": (time.time() - t1) / n_iterations,
            "Cumulative Model Updates": self.cumulative_model_updates,
            "Policy Entropy": mean_entropy,
            "Mean KL Divergence": mean_divergence,
            "Value Function Loss": mean_val_loss,
            "SB3 Clip Fraction": mean_clip,
            "Policy Update Magnitude": policy_update_magnitude,
            "Value Function Update Magnitude": critic_update_magnitude,
        }

        return report

    def save_to(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(folder_path, "PPO_POLICY.pt"))
        torch.save(
            self.value_net.state_dict(), os.path.join(folder_path, "PPO_VALUE_NET.pt")
        )
        torch.save(
            self.policy_optimizer.state_dict(),
            os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"),
        )
        torch.save(
            self.value_optimizer.state_dict(),
            os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"),
        )

    def load_from(self, folder_path):
        assert os.path.exists(folder_path), "PPO LEARNER CANNOT FIND FOLDER {}".format(
            folder_path
        )

        self.policy.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_POLICY.pt"))
        )
        self.value_net.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_VALUE_NET.pt"))
        )
        self.policy_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"))
        )
        self.value_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"))
        )
