import os
import time

import numpy as np
import torch
import torch.nn as nn

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

        if batch_size % mini_batch_size != 0:
            raise ValueError("MINIBATCH SIZE MUST BE AN INTEGER MULTIPLE OF BATCH SIZE")

        # Initialize policy based on type
        if policy_type == 2:
            self.policy = ContinuousPolicy(
                obs_space_size,
                act_space_size * 2,
                policy_layer_sizes,
                device,
                var_min=continuous_var_range[0],
                var_max=continuous_var_range[1],
            )
        elif policy_type == 1:
            self.policy = MultiDiscreteFF(obs_space_size, policy_layer_sizes, device)
        else:
            self.policy = DiscreteFF(
                obs_space_size, act_space_size, policy_layer_sizes, device
            )

        self.value_net = ValueEstimator(obs_space_size, critic_layer_sizes, device)

        # Move models to the specified device
        self.policy.to(device)
        self.value_net.to(device)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=critic_lr
        )
        self.value_loss_fn = nn.MSELoss()

        # Parameter count logging
        self.log_parameter_count()

        # Training configurations
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.mini_batch_size = mini_batch_size
        self.cumulative_model_updates = 0

    def log_parameter_count(self):
        policy_params_count = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        critic_params_count = sum(
            p.numel() for p in self.value_net.parameters() if p.requires_grad
        )
        total_parameters = policy_params_count + critic_params_count

        print(
            f"Trainable Parameters: Policy: {policy_params_count}, Critic: {critic_params_count}, Total: {total_parameters}"
        )

    def learn(self, exp):
        mean_entropy, mean_divergence, mean_val_loss, clip_fractions = 0, 0, 0, []
        mean_policy_loss, mean_value_loss, mean_entropy_loss = 0, 0, 0
        total_advantages, total_rewards = 0, 0

        policy_before = (
            torch.nn.utils.parameters_to_vector(self.policy.parameters()).detach().cpu()
        )
        critic_before = (
            torch.nn.utils.parameters_to_vector(self.value_net.parameters())
            .detach()
            .cpu()
        )

        t1 = time.time()
        epoch_durations = []
        total_batches = 0

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            batches = exp.get_all_batches_shuffled(self.batch_size)
            for batch in batches:
                total_batches += 1
                # Move data to the device and preprocess
                batch = [item.to(self.device) for item in batch]
                (
                    batch_acts,
                    batch_old_probs,
                    batch_obs,
                    batch_target_values,
                    batch_advantages,
                ) = batch

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                for start in range(0, self.batch_size, self.mini_batch_size):
                    end = start + self.mini_batch_size

                    minibatch_data = [item[start:end] for item in batch]
                    acts, old_probs, obs, target_values, advantages = minibatch_data

                    vals = self.value_net(obs).squeeze()

                    log_probs, entropy = self.policy.get_backprop_data(obs, acts)

                    ratio = torch.exp(log_probs - old_probs)
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                    )

                    # Calculate losses
                    policy_loss = -torch.min(
                        ratio * advantages, clipped_ratio * advantages
                    ).mean()
                    value_loss = self.value_loss_fn(vals, target_values)
                    entropy_loss = -entropy.mean()

                    total_loss = policy_loss + self.ent_coef * entropy_loss + value_loss
                    total_loss.backward()

                    mean_policy_loss += policy_loss.item()
                    mean_value_loss += value_loss.item()
                    mean_entropy_loss += entropy_loss.item()

                    mean_val_loss += value_loss.item()
                    mean_entropy += entropy.mean().item()
                    mean_divergence += (
                        ((ratio - 1) - (log_probs - old_probs)).mean().item()
                    )
                    clip_fractions.append(
                        (torch.abs(ratio - 1) > self.clip_range).float().mean().item()
                    )

                    total_advantages += advantages.mean().item()
                    total_rewards += target_values.mean().item()

                # Gradient clipping and optimization step
                grad_norm_policy = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 0.5
                ).item()
                grad_norm_value = torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), 0.5
                ).item()
                self.policy_optimizer.step()
                self.value_optimizer.step()

            epoch_duration = time.time() - epoch_start
            epoch_durations.append(epoch_duration)

        # Update and report generation
        policy_after = (
            torch.nn.utils.parameters_to_vector(self.policy.parameters()).detach().cpu()
        )
        critic_after = (
            torch.nn.utils.parameters_to_vector(self.value_net.parameters())
            .detach()
            .cpu()
        )
        policy_update_magnitude = (policy_before - policy_after).norm().item()
        critic_update_magnitude = (critic_before - critic_after).norm().item()

        total_batches *= self.n_epochs
        mean_entropy /= total_batches
        mean_divergence /= total_batches
        mean_val_loss /= total_batches
        mean_policy_loss /= total_batches
        mean_value_loss /= total_batches
        mean_entropy_loss /= total_batches
        total_advantages /= total_batches
        total_rewards /= total_batches
        mean_clip = np.mean(clip_fractions) if clip_fractions else 0

        self.cumulative_model_updates += total_batches

        report = {
            "PPO Batch Consumption Time": (time.time() - t1) / total_batches,
            "Cumulative Model Updates": self.cumulative_model_updates,
            "Policy Entropy": mean_entropy,
            "Mean KL Divergence": mean_divergence,
            "Policy Loss": mean_policy_loss,
            "Value Function Loss": mean_val_loss,
            "Entropy Loss": mean_entropy_loss,
            "Policy Gradient Norm": grad_norm_policy,
            "Value Gradient Norm": grad_norm_value,
            "Episode Mean Advantage": total_advantages,
            "Episode Mean Reward": total_rewards,
            "SB3 Clip Fraction": mean_clip,
            "Policy Update Magnitude": policy_update_magnitude,
            "Value Function Update Magnitude": critic_update_magnitude,
            "Mean Epoch Duration": np.mean(epoch_durations),
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
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"PPO LEARNER CANNOT FIND FOLDER {folder_path}")

        self.policy.load_state_dict(
            torch.load(
                os.path.join(folder_path, "PPO_POLICY.pt"), map_location=self.device
            )
        )
        self.value_net.load_state_dict(
            torch.load(
                os.path.join(folder_path, "PPO_VALUE_NET.pt"), map_location=self.device
            )
        )
        self.policy_optimizer.load_state_dict(
            torch.load(
                os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"),
                map_location=self.device,
            )
        )
        self.value_optimizer.load_state_dict(
            torch.load(
                os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"),
                map_location=self.device,
            )
        )
