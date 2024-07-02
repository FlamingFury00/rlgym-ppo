import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

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

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=critic_lr
        )
        self.value_loss_fn = nn.MSELoss()

        # Learning rate schedulers
        self.policy_scheduler = StepLR(
            self.policy_optimizer, step_size=1000, gamma=0.95
        )
        self.value_scheduler = StepLR(self.value_optimizer, step_size=1000, gamma=0.95)

        # Mixed precision training
        self.scaler = GradScaler()

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.cumulative_model_updates = 0

        # JIT compile the models
        # self.policy = torch.jit.script(self.policy)
        # self.value_net = torch.jit.script(self.value_net)

        # Display parameter counts and learning rates
        self._display_model_info(policy_lr, critic_lr)

    def _display_model_info(self, policy_lr, critic_lr):
        policy_params = self.policy.parameters()
        critic_params = self.value_net.parameters()

        trainable_policy_parameters = filter(lambda p: p.requires_grad, policy_params)
        policy_params_count = sum(p.numel() for p in trainable_policy_parameters)

        trainable_critic_parameters = filter(lambda p: p.requires_grad, critic_params)
        critic_params_count = sum(p.numel() for p in trainable_critic_parameters)

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

    def learn(self, actions, log_probs, states, values, advantages, weights):
        n_iterations = 0
        n_minibatch_iterations = 0
        mean_entropy = 0
        mean_divergence = 0
        mean_val_loss = 0
        clip_fractions = []

        policy_before = torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        ).cpu()
        critic_before = torch.nn.utils.parameters_to_vector(
            self.value_net.parameters()
        ).cpu()

        t1 = time.time()
        for epoch in range(self.n_epochs):
            # Shuffle the data
            permutation = torch.randperm(self.batch_size)
            actions = actions[permutation]
            log_probs = log_probs[permutation]
            states = states[permutation]
            values = values[permutation]
            advantages = advantages[permutation]
            weights = weights[permutation]

            for minibatch_slice in range(0, self.batch_size, self.mini_batch_size):
                start = minibatch_slice
                stop = start + self.mini_batch_size

                mb_actions = actions[start:stop].to(self.device)
                mb_log_probs = log_probs[start:stop].to(self.device)
                mb_states = states[start:stop].to(self.device)
                mb_values = values[start:stop].to(self.device)
                mb_advantages = advantages[start:stop].to(self.device)
                mb_weights = weights[start:stop].to(self.device)

                with autocast(False):
                    new_values = self.value_net(mb_states).view_as(mb_values)
                    new_log_probs, entropy = self.policy.get_backprop_data(
                        mb_states, mb_actions
                    )
                    new_log_probs = new_log_probs.view_as(mb_log_probs)

                    ratio = torch.exp(new_log_probs - mb_log_probs)
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                    )

                    with torch.no_grad():
                        log_ratio = new_log_probs - mb_log_probs
                        kl = (torch.exp(log_ratio) - 1) - log_ratio
                        kl = kl.mean().item()
                        clip_fraction = torch.mean(
                            (torch.abs(ratio - 1) > self.clip_range).float()
                        ).item()

                    policy_loss = -torch.min(
                        ratio * mb_advantages, clipped * mb_advantages
                    )
                    policy_loss = (policy_loss * mb_weights).mean()
                    value_loss = self.value_loss_fn(new_values, mb_values)
                    value_loss = (value_loss * mb_weights).mean()
                    ppo_loss = (
                        (policy_loss - entropy * self.ent_coef)
                        * self.mini_batch_size
                        / self.batch_size
                    )

                self.scaler.scale(ppo_loss).backward()
                self.scaler.scale(value_loss).backward()

                mean_val_loss += value_loss.item()
                mean_divergence += kl
                mean_entropy += entropy.item()
                clip_fractions.append(clip_fraction)
                n_minibatch_iterations += 1

            self.scaler.unscale_(self.policy_optimizer)
            self.scaler.unscale_(self.value_optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)

            self.scaler.step(self.policy_optimizer)
            self.scaler.step(self.value_optimizer)
            self.scaler.update()

            self.policy_scheduler.step()
            self.value_scheduler.step()

            n_iterations += 1

        if n_iterations == 0:
            n_iterations = 1
        if n_minibatch_iterations == 0:
            n_minibatch_iterations = 1

        mean_entropy /= n_minibatch_iterations
        mean_divergence /= n_minibatch_iterations
        mean_val_loss /= n_minibatch_iterations
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
            "Policy Learning Rate": self.policy_scheduler.get_last_lr()[0],
            "Value Learning Rate": self.value_scheduler.get_last_lr()[0],
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
        torch.save(
            self.policy_scheduler.state_dict(),
            os.path.join(folder_path, "PPO_POLICY_SCHEDULER.pt"),
        )
        torch.save(
            self.value_scheduler.state_dict(),
            os.path.join(folder_path, "PPO_VALUE_SCHEDULER.pt"),
        )

    def load_from(self, folder_path):
        assert os.path.exists(
            folder_path
        ), f"PPO LEARNER CANNOT FIND FOLDER {folder_path}"

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

        # Load scheduler states
        policy_scheduler_state = torch.load(
            os.path.join(folder_path, "PPO_POLICY_SCHEDULER.pt")
        )
        value_scheduler_state = torch.load(
            os.path.join(folder_path, "PPO_VALUE_SCHEDULER.pt")
        )

        self.policy_scheduler.load_state_dict(policy_scheduler_state)
        self.value_scheduler.load_state_dict(value_scheduler_state)

        # Set the learning rates to the last used learning rates
        last_policy_lr = self.policy_scheduler.get_last_lr()[0]
        last_value_lr = self.value_scheduler.get_last_lr()[0]

        for param_group in self.policy_optimizer.param_groups:
            param_group["lr"] = last_policy_lr

        for param_group in self.value_optimizer.param_groups:
            param_group["lr"] = last_value_lr

        print(
            f"Loaded checkpoint. Set policy LR to {last_policy_lr} and value LR to {last_value_lr}"
        )
