import json
import os
import random
import shutil
import time
from typing import Tuple, Union

import numpy as np
import torch
import wandb
from wandb.sdk.wandb_run import Run  # Corrected import

from rlgym_ppo.batched_agents import BatchedAgentManager
from rlgym_ppo.ppo import ExperienceBuffer, PPOLearner
from rlgym_ppo.util import KBHit, WelfordRunningStat, reporting, torch_functions


class Learner(object):
    def __init__(
        # fmt: off
            self,
            env_create_function,
            metrics_logger=None,
            n_proc: int = 8,
            min_inference_size: int = 80,
            render: bool = False,
            render_delay: float = 0,

            timestep_limit: int = 5_000_000_000,
            exp_buffer_size: int = 100000,
            ts_per_iteration: int = 50000,
            standardize_returns: bool = True,
            standardize_obs: bool = True,
            max_returns_per_stats_increment: int = 150,
            steps_per_obs_stats_increment: int = 5,

            policy_layer_sizes: Tuple[int, ...] = (256, 256, 256),
            critic_layer_sizes: Tuple[int, ...] = (256, 256, 256),
            continuous_var_range: Tuple[float, ...] = (0.1, 1.0),

            ppo_epochs: int = 10,
            ppo_batch_size: int = 50000,
            ppo_minibatch_size: Union[int, None] = None,
            ppo_ent_coef: float = 0.01,  # Updated default entropy coefficient
            ppo_clip_range: float = 0.1,  # Updated default clip range

            gae_lambda: float = 0.95,
            gae_gamma: float = 0.99,
            policy_lr: float = 1e-4,  # Updated default policy learning rate
            critic_lr: float = 1e-4,  # Updated default critic learning rate

            log_to_wandb: bool = False,
            load_wandb: bool = True,
            wandb_run: Union[Run, None] = None,
            wandb_project_name: Union[str, None] = None,
            wandb_group_name: Union[str, None] = None,
            wandb_run_name: Union[str, None] = None,

            checkpoints_save_folder: Union[str, None] = None,
            add_unix_timestamp: bool = True,
            checkpoint_load_folder: Union[str, None] = None, # "latest" loads latest checkpoint
            save_every_ts: int = 1_000_000,

            instance_launch_delay: Union[float, None] = None,
            random_seed: int = 123,
            n_checkpoints_to_keep: int = 5,
            shm_buffer_size: int = 8192,
            device: str = "auto"
    ):

        if env_create_function is None:
            raise ValueError("The `env_create_function` parameter must be provided to initialize RLGym-PPO.")

        if checkpoints_save_folder is None:
            checkpoints_save_folder = os.path.join(
                "data", "checkpoints", "rlgym-ppo-run"
            )

        self.add_unix_timestamp = add_unix_timestamp
        if add_unix_timestamp:
            checkpoints_save_folder = f"{checkpoints_save_folder}-{time.time_ns()}"

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.n_checkpoints_to_keep = n_checkpoints_to_keep
        self.checkpoints_save_folder = checkpoints_save_folder
        self.max_returns_per_stats_increment = max_returns_per_stats_increment
        self.metrics_logger = metrics_logger
        self.standardize_returns = standardize_returns
        self.save_every_ts = save_every_ts
        self.ts_since_last_save = 0

        if device in {"auto", "gpu"} and torch.cuda.is_available():
            self.device = "cuda:0"
            torch.cuda.set_device(self.device)  # Ensure the correct GPU is selected
            torch.backends.cudnn.benchmark = True
        elif device == "auto" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        print(f"Using device {self.device}")
        self.exp_buffer_size = exp_buffer_size
        self.timestep_limit = timestep_limit
        self.ts_per_epoch = ts_per_iteration
        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gamma
        self.return_stats = WelfordRunningStat(1)
        self.epoch = 0

        # Initialize ExperienceBuffer directly on the target device if possible
        self.experience_buffer = ExperienceBuffer(
            self.exp_buffer_size, seed=random_seed, device=self.device
        )

        print("Initializing processes...")
        collect_metrics_fn = (
            None if metrics_logger is None else self.metrics_logger.collect_metrics
        )
        self.agent = BatchedAgentManager(
            None,
            min_inference_size=min_inference_size,
            seed=random_seed,
            standardize_obs=standardize_obs,
            steps_per_obs_stats_increment=steps_per_obs_stats_increment,
        )
        obs_space_size, act_space_size, action_space_type = self.agent.init_processes(
            n_processes=n_proc,
            build_env_fn=env_create_function,
            collect_metrics_fn=collect_metrics_fn,
            spawn_delay=instance_launch_delay,
            render=render,
            render_delay=render_delay,
            shm_buffer_size=shm_buffer_size,
        )
        obs_space_size = np.prod(obs_space_size)
        print("Initializing PPO...")
        if ppo_minibatch_size is None:
            ppo_minibatch_size = ppo_batch_size

        self.ppo_learner = PPOLearner(
            obs_space_size,
            act_space_size,
            device=self.device,
            batch_size=ppo_batch_size,
            mini_batch_size=ppo_minibatch_size,
            n_epochs=ppo_epochs,
            continuous_var_range=continuous_var_range,
            policy_type=action_space_type,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            clip_range=ppo_clip_range,
            ent_coef=ppo_ent_coef,
        )

        self.agent.policy = self.ppo_learner.policy

        self.config = {
            "n_proc": n_proc,
            "min_inference_size": min_inference_size,
            "timestep_limit": timestep_limit,
            "exp_buffer_size": exp_buffer_size,
            "ts_per_iteration": ts_per_iteration,
            "standardize_returns": standardize_returns,
            "standardize_obs": standardize_obs,
            "policy_layer_sizes": policy_layer_sizes,
            "critic_layer_sizes": critic_layer_sizes,
            "ppo_epochs": ppo_epochs,
            "ppo_batch_size": ppo_batch_size,
            "ppo_minibatch_size": ppo_minibatch_size,
            "ppo_ent_coef": ppo_ent_coef,
            "ppo_clip_range": ppo_clip_range,
            "gae_lambda": gae_lambda,
            "gae_gamma": gae_gamma,
            "policy_lr": policy_lr,
            "critic_lr": critic_lr,
            "shm_buffer_size": shm_buffer_size,
        }

        self.wandb_run = wandb_run
        wandb_loaded = checkpoint_load_folder is not None and self.load(
            checkpoint_load_folder, load_wandb, policy_lr, critic_lr
        )

        if log_to_wandb and self.wandb_run is None and not wandb_loaded:
            project = "rlgym-ppo" if wandb_project_name is None else wandb_project_name
            group = "unnamed-runs" if wandb_group_name is None else wandb_group_name
            run_name = "rlgym-ppo-run" if wandb_run_name is None else wandb_run_name
            print("Attempting to create new wandb run...")
            self.wandb_run = wandb.init(
                project=project,
                group=group,
                config=self.config,
                name=run_name,
                reinit=True,
                allow_val_change=True,
            )
            print("Created new wandb run!", self.wandb_run.id)
        print("Learner successfully initialized!")

    def learn(self) -> None:
        """
        Wraps the `_learn` function in a try/catch/finally block to ensure safe execution and error handling.

        This function handles unexpected errors during training, attempts to save the current state, and performs
        cleanup operations before exiting.

        :return: None
        """

    def _learn(self) -> None:
        """
        The main learning loop for the PPO algorithm.

        This function collects timesteps, processes experiences, and performs PPO updates. It also handles
        logging, checkpointing, and user interactions (e.g., pausing or quitting).

        :return: None
        """
        kb = KBHit()
        print("Press (p) to pause, (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\\n")

        while self.agent.cumulative_timesteps < self.timestep_limit:
            try:
                epoch_start = time.perf_counter()
                report = {}

                # Collect timesteps
                experience, collected_metrics, steps_collected, collection_time = self.agent.collect_timesteps(self.ts_per_epoch)

                # Report metrics if logger is provided
                if self.metrics_logger is not None:
                    self.metrics_logger.report_metrics(
                        collected_metrics, self.wandb_run, self.agent.cumulative_timesteps
                    )

                # Add new experience to the buffer
                add_exp_start_time = time.perf_counter()
                self.add_new_experience(experience)
                add_exp_time = time.perf_counter() - add_exp_start_time

                # Perform PPO updates
                ppo_start_time = time.perf_counter()
                ppo_report = self.ppo_learner.learn(self.experience_buffer)
                ppo_time = time.perf_counter() - ppo_start_time

                # Calculate epoch time
                epoch_stop = time.perf_counter()
                epoch_time = epoch_stop - epoch_start

                # Update report with metrics
                report.update(ppo_report)
                report["Cumulative Timesteps"] = self.agent.cumulative_timesteps
                report["Total Iteration Time"] = epoch_time
                report["Timesteps Collected"] = steps_collected
                report["Timestep Collection Time"] = collection_time
                report["Experience Processing Time"] = add_exp_time
                report["PPO Learning Time"] = ppo_time
                report["Timestep Consumption Time"] = epoch_time - collection_time - add_exp_time
                report["Collected Steps per Second"] = steps_collected / collection_time
                report["Overall Steps per Second"] = steps_collected / epoch_time

                self.ts_since_last_save += steps_collected
                report["Policy Reward"] = self.agent.average_reward if self.agent.average_reward is not None else np.nan

                # Log metrics
                reporting.report_metrics(loggable_metrics=report, debug_metrics=None, wandb_run=self.wandb_run)

                # Clear temporary reports
                report.clear()
                ppo_report.clear()

                # Free GPU memory if applicable
                if "cuda" in self.device:
                    torch.cuda.empty_cache()

                # Handle keyboard input
                if kb.kbhit():
                    c = kb.getch()
                    if c == "p":
                        print("Paused, press any key to resume")
                        while True:
                            if kb.kbhit():
                                break
                    if c in ("c", "q"):
                        self.save(self.agent.cumulative_timesteps)
                    if c == "q":
                        return
                    if c in ("c", "p"):
                        print("Resuming...\\n")

                # Save checkpoint if necessary
                if self.ts_since_last_save >= self.save_every_ts:
                    self.save(self.agent.cumulative_timesteps)
                    self.ts_since_last_save = 0

                self.epoch += 1

            except Exception as e:
                print(f"Error during learning loop: {e}")
                self.save(self.agent.cumulative_timesteps)
                break

    def update_learning_rate(self, new_policy_lr: Union[float, None] = None, new_critic_lr: Union[float, None] = None) -> bool:
        """
        Updates the learning rates for the policy and critic optimizers.

        :param new_policy_lr: The new learning rate for the policy optimizer. If None, the policy learning rate remains unchanged.
        :param new_critic_lr: The new learning rate for the critic optimizer. If None, the critic learning rate remains unchanged.
        :return: True if any learning rate was updated, False otherwise.
        """
        updated = False

        if new_policy_lr is not None:
            for param_group in self.ppo_learner.policy_optimizer.param_groups:
                param_group['lr'] = new_policy_lr
            updated = True

        if new_critic_lr is not None:
            for param_group in self.ppo_learner.critic_optimizer.param_groups:
                param_group['lr'] = new_critic_lr
            updated = True

        return updated

                n_iterations += 1

            # Step learning rate schedulers
            self.policy_scheduler.step()
            self.value_scheduler.step()