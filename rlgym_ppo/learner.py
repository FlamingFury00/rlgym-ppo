"""
File: learner.py
Author: Matthew Allen

Description:
The primary algorithm file. The Learner object coordinates timesteps from the workers
and sends them to PPO, keeps track of the misc. variables and statistics for logging,
reports to wandb and the console, and handles checkpointing.
"""

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
            ppo_ent_coef: float = 0.005,
            ppo_clip_range: float = 0.2,

            gae_lambda: float = 0.95,
            gae_gamma: float = 0.99,
            policy_lr: float = 3e-4,
            critic_lr: float = 3e-4,

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

        assert (
            env_create_function is not None
        ), "MUST PROVIDE A FUNCTION TO CREATE RLGYM FUNCTIONS TO INITIALIZE RLGYM-PPO"

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

    def update_learning_rate(self, new_policy_lr=None, new_critic_lr=None):
        updated = False
        if (
            new_policy_lr is not None
            and self.ppo_learner.policy_optimizer.param_groups[0]["lr"] != new_policy_lr
        ):
            self.policy_lr = new_policy_lr
            for param_group in self.ppo_learner.policy_optimizer.param_groups:
                param_group["lr"] = new_policy_lr
            print(f"New policy learning rate: {new_policy_lr}")
            updated = True

        if (
            new_critic_lr is not None
            and self.ppo_learner.value_optimizer.param_groups[0]["lr"] != new_critic_lr
        ):
            self.critic_lr = new_critic_lr
            for param_group in self.ppo_learner.value_optimizer.param_groups:
                param_group["lr"] = new_critic_lr
            print(f"New critic learning rate: {new_critic_lr}")
            updated = True

        return updated

    def learn(self):
        """
        Function to wrap the _learn function in a try/catch/finally
        block to ensure safe execution and error handling.
        :return: None
        """
        try:
            self._learn()
        except Exception:
            import traceback

            print("\\n\\nLEARNING LOOP ENCOUNTERED AN ERROR\\n")
            traceback.print_exc()

            try:
                self.save(self.agent.cumulative_timesteps)
            except Exception:
                print("FAILED TO SAVE ON EXIT")

        finally:
            self.cleanup()

    def _learn(self):
        """
        Learning function. This is where the magic happens.
        :return: None
        """

        kb = KBHit()
        print(
            "Press (p) to pause (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\\n"
        )

        while self.agent.cumulative_timesteps < self.timestep_limit:
            epoch_start = time.perf_counter()
            report = {}

            # Collect timesteps and directly move the experience to the target device
            experience, collected_metrics, steps_collected, collection_time = (
                self.agent.collect_timesteps(self.ts_per_epoch)
            )

            if self.metrics_logger is not None:
                self.metrics_logger.report_metrics(
                    collected_metrics, self.wandb_run, self.agent.cumulative_timesteps
                )

            # Add new experience and compute RL quantities directly on the target device
            add_exp_start_time = time.perf_counter()
            self.add_new_experience(experience)
            add_exp_time = time.perf_counter() - add_exp_start_time

            # Learn from the experience buffer
            ppo_start_time = time.perf_counter()
            ppo_report = self.ppo_learner.learn(self.experience_buffer)
            ppo_time = time.perf_counter() - ppo_start_time

            epoch_stop = time.perf_counter()
            epoch_time = epoch_stop - epoch_start

            report.update(ppo_report)
            report["Cumulative Timesteps"] = self.agent.cumulative_timesteps
            report["Total Iteration Time"] = epoch_time
            report["Timesteps Collected"] = steps_collected
            report["Timestep Collection Time"] = collection_time
            report["Experience Processing Time"] = add_exp_time
            report["PPO Learning Time"] = ppo_time
            report["Timestep Consumption Time"] = (
                epoch_time - collection_time - add_exp_time
            )  # Correct calculation
            report["Collected Steps per Second"] = steps_collected / collection_time
            report["Overall Steps per Second"] = steps_collected / epoch_time

            self.ts_since_last_save += steps_collected
            report["Policy Reward"] = (
                self.agent.average_reward
                if self.agent.average_reward is not None
                else np.nan
            )

            reporting.report_metrics(
                loggable_metrics=report, debug_metrics=None, wandb_run=self.wandb_run
            )

            report.clear()
            ppo_report.clear()

            if "cuda" in self.device:
                torch.cuda.empty_cache()

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

            if self.ts_since_last_save >= self.save_every_ts:
                self.save(self.agent.cumulative_timesteps)
                self.ts_since_last_save = 0

            self.epoch += 1

    @torch.no_grad()
    def add_new_experience(self, experience):
        states, actions, log_probs, rewards, next_states, dones, truncated = experience
        value_net = self.ppo_learner.value_net

        # Move data to the target device once here
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        truncated_tensor = torch.as_tensor(truncated, dtype=torch.float32, device=self.device)

        # Construct input for value function estimator
        val_inp = torch.cat((states_tensor, next_states_tensor[-1].unsqueeze(0)), dim=0) if next_states_tensor.numel() > 0 else states_tensor

        # Predict expected returns
        val_preds = value_net(val_inp).flatten()

        # Compute GAE directly on GPU
        ret_std = self.return_stats.std[0] if self.standardize_returns else None
        value_targets, advantages, returns = torch_functions.compute_gae_torch(
            rewards_tensor, dones_tensor, truncated_tensor, val_preds, 
            gamma=self.gae_gamma, lmbda=self.gae_lambda, return_std=ret_std
        )

        # Pre-allocate tensors for value_targets and advantages
        value_targets = torch.as_tensor(value_targets, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        if self.standardize_returns:
            n_to_increment = min(self.max_returns_per_stats_increment, len(returns))
            self.return_stats.increment(returns[:n_to_increment], n_to_increment)

        # Submit experience to the buffer
        self.experience_buffer.submit_experience(
            states,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            truncated,
            value_targets.cpu().numpy(),  # Convert to numpy for buffer compatibility
            advantages.cpu().numpy(),
        )

    def save(self, cumulative_timesteps):
        """
        Function to save a checkpoint.
        :param cumulative_timesteps: Number of timesteps that have passed so far in the
        learning algorithm.
        :return: None
        """

        # Make the file path to which the checkpoint will be saved
        folder_path = os.path.join(
            self.checkpoints_save_folder, str(cumulative_timesteps)
        )
        os.makedirs(folder_path, exist_ok=True)

        # Check to see if we've run out of checkpoint space and remove the oldest
        # checkpoints
        print(f"Saving checkpoint {cumulative_timesteps}...")
        existing_checkpoints = [
            int(arg) for arg in os.listdir(self.checkpoints_save_folder)
        ]
        if len(existing_checkpoints) > self.n_checkpoints_to_keep:
            existing_checkpoints.sort()
            for checkpoint_name in existing_checkpoints[: -self.n_checkpoints_to_keep]:
                shutil.rmtree(
                    os.path.join(self.checkpoints_save_folder, str(checkpoint_name))
                )

        os.makedirs(folder_path, exist_ok=True)

        # Save all the things that need saving.
        self.ppo_learner.save_to(folder_path)

        book_keeping_vars = {
            "cumulative_timesteps": self.agent.cumulative_timesteps,
            "cumulative_model_updates": self.ppo_learner.cumulative_model_updates,
            "policy_average_reward": self.agent.average_reward,
            "epoch": self.epoch,
            "ts_since_last_save": self.ts_since_last_save,
            "reward_running_stats": self.return_stats.to_json(),
        }
        if self.agent.standardize_obs:
            book_keeping_vars["obs_running_stats"] = self.agent.obs_stats.to_json()
        if self.standardize_returns:
            book_keeping_vars["reward_running_stats"] = self.return_stats.to_json()

        if self.wandb_run is not None:
            book_keeping_vars["wandb_run_id"] = self.wandb_run.id
            book_keeping_vars["wandb_project"] = self.wandb_run.project
            book_keeping_vars["wandb_entity"] = self.wandb_run.entity
            book_keeping_vars["wandb_group"] = self.wandb_run.group
            book_keeping_vars["wandb_config"] = self.wandb_run.config.as_dict()

        book_keeping_table_path = os.path.join(folder_path, "BOOK_KEEPING_VARS.json")
        with open(book_keeping_table_path, "w") as f:
            json.dump(book_keeping_vars, f, indent=4)

        print(f"Checkpoint {cumulative_timesteps} saved!\\n")

    def load(self, folder_path, load_wandb, new_policy_lr=None, new_critic_lr=None):
        """
        Function to load the learning algorithm from a checkpoint.

        :param folder_path: Path to the checkpoint folder that will be loaded.
        :param load_wandb: Whether to resume an existing weights and biases run that
        was saved with the checkpoint being loaded.
        :return: None
        """

        # Make sure the folder exists.
        assert os.path.exists(folder_path), f"UNABLE TO LOCATE FOLDER {folder_path}"
        print(f"Loading from checkpoint at {folder_path}")

        # Load stuff.
        self.ppo_learner.load_from(folder_path)

        wandb_loaded = False
        with open(os.path.join(folder_path, "BOOK_KEEPING_VARS.json"), "r") as f:
            book_keeping_vars = dict(json.load(f))
            self.agent.cumulative_timesteps = book_keeping_vars["cumulative_timesteps"]
            self.agent.average_reward = book_keeping_vars["policy_average_reward"]
            self.ppo_learner.cumulative_model_updates = book_keeping_vars[
                "cumulative_model_updates"
            ]
            self.return_stats.from_json(book_keeping_vars["reward_running_stats"])

            if (
                self.agent.standardize_obs
                and "obs_running_stats" in book_keeping_vars.keys()
            ):
                self.agent.obs_stats = WelfordRunningStat(1)
                self.agent.obs_stats.from_json(book_keeping_vars["obs_running_stats"])
            if (
                self.standardize_returns
                and "reward_running_stats" in book_keeping_vars.keys()
            ):
                self.return_stats.from_json(book_keeping_vars["reward_running_stats"])

            self.epoch = book_keeping_vars["epoch"]

            # Update learning rates if new values are provided
            if new_policy_lr is not None or new_critic_lr is not None:
                self.update_learning_rate(new_policy_lr, new_critic_lr)

            # check here for backwards compatibility

            if "wandb_run_id" in book_keeping_vars and load_wandb:
                self.wandb_run = wandb.init(
                    settings=wandb.Settings(start_method="spawn"),
                    entity=book_keeping_vars["wandb_entity"],
                    project=book_keeping_vars["wandb_project"],
                    group=book_keeping_vars["wandb_group"],
                    id=book_keeping_vars["wandb_run_id"],
                    config=book_keeping_vars["wandb_config"],
                    resume="allow",
                    reinit=True,
                )
                wandb_loaded = True

        print("Checkpoint loaded!")
        return wandb_loaded

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """

        if self.wandb_run is not None:
            self.wandb_run.finish()
        if type(self.agent) is BatchedAgentManager:
            self.agent.cleanup()
        self.experience_buffer.clear()