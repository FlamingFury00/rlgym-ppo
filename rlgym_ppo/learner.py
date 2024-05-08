import json
import logging
import os
import random
import shutil
import time
from typing import Callable, Tuple, Union

import numpy as np
import torch
import wandb
from rlgym_sim import gym
from wandb.wandb_run import Run

from rlgym_ppo.batched_agents import BatchedAgentManager
from rlgym_ppo.ppo import ExperienceBuffer, PPOLearner
from rlgym_ppo.util import KBHit, WelfordRunningStat, reporting, torch_functions

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Learner(object):
    def __init__(
        self,
        env_create_function: Callable[..., gym.Gym],
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
        checkpoint_load_folder: Union[str, None] = None,
        save_every_ts: int = 1_000_000,
        instance_launch_delay: Union[float, None] = None,
        random_seed: int = 123,
        n_checkpoints_to_keep: int = 5,
        shm_buffer_size: int = 8192,
        device: str = "auto",
    ):

        assert (
            env_create_function is not None
        ), "Must provide a function to create RLGYM functions to initialize RLGYM-PPO"

        self.checkpoints_save_folder = checkpoints_save_folder or os.path.join(
            "data", "checkpoints", "rlgym-ppo-run"
        )
        if add_unix_timestamp:
            self.checkpoints_save_folder += f"-{time.time_ns()}"

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.n_checkpoints_to_keep = n_checkpoints_to_keep
        self.metrics_logger = metrics_logger
        self.standardize_returns = standardize_returns
        self.save_every_ts = save_every_ts
        self.ts_since_last_save = 0

        self.device = self._setup_device(device)
        logger.info(f"Using device {self.device}")

        self.timestep_limit = timestep_limit
        self.ts_per_epoch = ts_per_iteration
        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gamma
        self.return_stats = WelfordRunningStat(1)
        self.epoch = 0

        self.policy_layer_sizes = policy_layer_sizes
        self.critic_layer_sizes = critic_layer_sizes

        self.policy_lr = policy_lr
        self.critic_lr = critic_lr

        self.max_returns_per_stats_increment = max_returns_per_stats_increment

        self.experience_buffer = ExperienceBuffer(
            exp_buffer_size, seed=random_seed, device="cpu"
        )
        logger.info("Initializing processes...")

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
            collect_metrics_fn=(
                None if metrics_logger is None else metrics_logger.collect_metrics
            ),
            spawn_delay=instance_launch_delay,
            render=render,
            render_delay=render_delay,
            shm_buffer_size=shm_buffer_size,
        )
        obs_space_size = np.prod(obs_space_size)

        logger.info("Initializing PPO...")
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

        self._initialize_wandb(
            log_to_wandb,
            load_wandb,
            wandb_run,
            wandb_project_name,
            wandb_group_name,
            wandb_run_name,
            checkpoint_load_folder,
            policy_lr,
            critic_lr,
        )

    def _setup_device(self, device):
        if device in {"auto", "gpu"} and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            return "cuda:0"
        elif device == "auto" and not torch.cuda.is_available():
            return "cpu"
        return device

    def _initialize_wandb(
        self,
        log_to_wandb,
        load_wandb,
        wandb_run,
        wandb_project_name,
        wandb_group_name,
        wandb_run_name,
        checkpoint_load_folder,
        policy_lr,
        critic_lr,
    ):
        self.wandb_run = wandb_run
        if checkpoint_load_folder:
            wandb_loaded = self.load(
                checkpoint_load_folder, load_wandb, policy_lr, critic_lr
            )
        else:
            wandb_loaded = False

        if log_to_wandb and self.wandb_run is None and not wandb_loaded:
            project = wandb_project_name or "rlgym-ppo"
            group = wandb_group_name or "unnamed-runs"
            run_name = wandb_run_name or "rlgym-ppo-run"
            logger.info("Attempting to create new wandb run...")
            self.wandb_run = wandb.init(
                project=project,
                group=group,
                config=self._get_config(),
                name=run_name,
                reinit=True,
            )
            logger.info(f"Created new wandb run! ID: {self.wandb_run.id}")
        logger.info("Learner successfully initialized!")

    def _get_config(self):
        return {
            "min_inference_size": self.agent.min_inference_size,
            "timestep_limit": self.timestep_limit,
            "ts_per_iteration": self.ts_per_epoch,
            "standardize_returns": self.standardize_returns,
            "standardize_obs": self.agent.standardize_obs,
            "policy_layer_sizes": self.policy_layer_sizes,
            "critic_layer_sizes": self.critic_layer_sizes,
            "ppo_epochs": self.ppo_learner.n_epochs,
            "ppo_batch_size": self.ppo_learner.batch_size,
            "ppo_minibatch_size": self.ppo_learner.mini_batch_size,
            "ppo_ent_coef": self.ppo_learner.ent_coef,
            "ppo_clip_range": self.ppo_learner.clip_range,
            "gae_lambda": self.gae_lambda,
            "gae_gamma": self.gae_gamma,
            "policy_lr": self.policy_lr,
            "critic_lr": self.critic_lr,
        }

    def update_learning_rate(self, new_policy_lr=None, new_critic_lr=None):
        if new_policy_lr is not None:
            self.ppo_learner.policy_lr = new_policy_lr
            for param_group in self.ppo_learner.policy_optimizer.param_groups:
                param_group["lr"] = new_policy_lr
            logger.info(f"New policy learning rate: {new_policy_lr}")

        if new_critic_lr is not None:
            self.ppo_learner.critic_lr = new_critic_lr
            for param_group in self.ppo_learner.value_optimizer.param_groups:
                param_group["lr"] = new_critic_lr
            logger.info(f"New critic learning rate: {new_critic_lr}")

    def learn(self):
        try:
            self._learn()
        except Exception:
            logger.error("LEARNING LOOP ENCOUNTERED AN ERROR", exc_info=True)
            try:
                self.save(self.agent.cumulative_timesteps)
            except Exception:
                logger.error("FAILED TO SAVE ON EXIT", exc_info=True)
        finally:
            self.cleanup()

    def _learn(self):
        kb = KBHit()
        logger.info(
            "Press (p) to pause, (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\n"
        )

        while self.agent.cumulative_timesteps < self.timestep_limit:
            epoch_start = time.perf_counter()
            report = {}

            experience, collected_metrics, steps_collected, collection_time = (
                self.agent.collect_timesteps(self.ts_per_epoch)
            )
            if self.metrics_logger is not None:
                self.metrics_logger.report_metrics(
                    collected_metrics, self.wandb_run, self.agent.cumulative_timesteps
                )

            self.add_new_experience(experience)
            ppo_report = self.ppo_learner.learn(self.experience_buffer)
            epoch_stop = time.perf_counter()
            epoch_time = epoch_stop - epoch_start

            report.update(ppo_report)
            report["Cumulative Timesteps"] = self.agent.cumulative_timesteps
            report["Total Iteration Time"] = epoch_time
            report["Timesteps Collected"] = steps_collected
            report["Timestep Collection Time"] = collection_time
            report["Timestep Consumption Time"] = epoch_time - collection_time
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

            if "cuda" in self.device:
                torch.cuda.empty_cache()

            if kb.kbhit():
                c = kb.getch()
                if c == "p":  # pause
                    logger.info("Paused, press any key to resume")
                    kb.wait_for_keypress()
                if c in ("c", "q"):  # checkpoint and possibly quit
                    self.save(self.agent.cumulative_timesteps)
                if c == "q":  # quit
                    break
                if c == "c":  # continue
                    logger.info("Resuming...\n")

            if self.ts_since_last_save >= self.save_every_ts:
                self.save(self.agent.cumulative_timesteps)
                self.ts_since_last_save = 0

            self.epoch += 1

    @torch.no_grad()
    def add_new_experience(self, experience):
        states, actions, log_probs, rewards, next_states, dones, truncated = experience
        val_inp = np.vstack([states, next_states[-1:]])
        val_preds = self.ppo_learner.value_net(val_inp).cpu().flatten().numpy()

        value_targets, advantages, returns = torch_functions.compute_gae(
            rewards,
            dones,
            truncated,
            val_preds,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
            return_std=self.return_stats.std[0] if self.standardize_returns else None,
        )

        if self.standardize_returns and len(returns) > 0:
            self.return_stats.increment(
                returns[: min(self.max_returns_per_stats_increment, len(returns))],
                min(self.max_returns_per_stats_increment, len(returns)),
            )

        self.experience_buffer.submit_experience(
            states,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            truncated,
            value_targets,
            advantages,
        )

    def save(self, cumulative_timesteps):
        folder_path = os.path.join(
            self.checkpoints_save_folder, str(cumulative_timesteps)
        )
        os.makedirs(folder_path, exist_ok=True)

        logger.info(f"Saving checkpoint {cumulative_timesteps}...")
        existing_checkpoints = sorted(
            int(name)
            for name in os.listdir(self.checkpoints_save_folder)
            if name.isdigit()
        )
        if len(existing_checkpoints) > self.n_checkpoints_to_keep:
            for old_checkpoint in existing_checkpoints[: -self.n_checkpoints_to_keep]:
                shutil.rmtree(
                    os.path.join(self.checkpoints_save_folder, str(old_checkpoint))
                )

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

        if self.wandb_run:
            book_keeping_vars.update(
                {
                    "wandb_run_id": self.wandb_run.id,
                    "wandb_project": self.wandb_run.project,
                    "wandb_entity": self.wandb_run.entity,
                    "wandb_group": self.wandb_run.group,
                    "wandb_config": self.wandb_run.config.as_dict(),
                }
            )

        with open(os.path.join(folder_path, "BOOK_KEEPING_VARS.json"), "w") as f:
            json.dump(book_keeping_vars, f, indent=4)

        logger.info(f"Checkpoint {cumulative_timesteps} saved!\n")

    def load(self, folder_path, load_wandb, new_policy_lr=None, new_critic_lr=None):
        assert os.path.exists(folder_path), f"Unable to locate folder {folder_path}"
        logger.info(f"Loading from checkpoint at {folder_path}")

        self.ppo_learner.load_from(folder_path)
        with open(os.path.join(folder_path, "BOOK_KEEPING_VARS.json"), "r") as f:
            book_keeping_vars = json.load(f)
            self.agent.cumulative_timesteps = book_keeping_vars.get(
                "cumulative_timesteps", 0
            )
            self.agent.average_reward = book_keeping_vars.get("policy_average_reward")
            self.ppo_learner.cumulative_model_updates = book_keeping_vars.get(
                "cumulative_model_updates", 0
            )
            self.return_stats.from_json(
                book_keeping_vars.get("reward_running_stats", "{}")
            )

            if self.agent.standardize_obs and "obs_running_stats" in book_keeping_vars:
                self.agent.obs_stats.from_json(book_keeping_vars["obs_running_stats"])
            if self.standardize_returns and "reward_running_stats" in book_keeping_vars:
                self.return_stats.from_json(book_keeping_vars["reward_running_stats"])

            self.epoch = book_keeping_vars.get("epoch", 0)

            if new_policy_lr is not None or new_critic_lr is not None:
                self.update_learning_rate(new_policy_lr, new_critic_lr)

            if "wandb_run_id" in book_keeping_vars and load_wandb:
                self.wandb_run = wandb.init(
                    settings=wandb.Settings(start_method="spawn"),
                    entity=book_keeping_vars["wandb_entity"],
                    project=book_keeping_vars["wandb_project"],
                    group=book_keeping_vars["wandb_group"],
                    id=book_keeping_vars["wandb_run_id"],
                    config=book_keeping_vars.get("wandb_config", {}),
                    resume="allow",
                    reinit=True,
                )
                return True
        logger.info("Checkpoint loaded!")
        return False

    def cleanup(self):
        if self.wandb_run:
            self.wandb_run.finish()
        if isinstance(self.agent, BatchedAgentManager):
            self.agent.cleanup()
        self.experience_buffer.clear()
