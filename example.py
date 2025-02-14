import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
        EventReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction

    # Whether to spawn opponents in the environment.
    spawn_opponents = True

    # Number of players per team.
    team_size = 1

    # Game tick rate and tick skip determine the simulation speed.
    game_tick_rate = 120  # Ticks per second in the simulation.
    tick_skip = 8  # Number of ticks to skip per action.

    # Timeout settings for terminating episodes.
    timeout_seconds = 10  # Maximum time (in seconds) before timeout.
    max_timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))  # Convert to ticks.

    action_parser = ContinuousAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    # Example 1: Default reward function combining multiple objectives.
    rewards_to_combine = (
        VelocityPlayerToBallReward(),
        VelocityBallToGoalReward(),
        EventReward(team_goal=1, concede=-1, demo=0.1)
    )
    reward_weights = (0.01, 0.1, 10.0)

    # Example 2: Alternative reward function focusing on ball control and scoring.
    # Uncomment the following lines to use this reward function instead.
    # rewards_to_combine = (
    #     EventReward(team_goal=2, concede=-2),  # Higher weight for scoring goals.
    #     VelocityBallToGoalReward(),  # Encourage moving the ball toward the goal.
    # )
    # reward_weights = (1.0, 0.5)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    # Example 1: Default observation builder with normalized coefficients.
    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL
    )

    # Example 2: Alternative observation builder with custom scaling.
    # Uncomment the following lines to use this observation builder instead.
    # obs_builder = DefaultObs(
    #     pos_coef=np.asarray([1 / 4096, 1 / 5120, 1 / 2044]),  # Custom field dimensions.
    #     ang_coef=1 / (2 * np.pi),  # Adjust angular normalization.
    #     lin_vel_coef=1 / 2300,  # Custom max speed normalization.
    #     ang_vel_coef=1 / 5.5  # Custom angular velocity normalization.
    # )

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    # Number of processes to run in parallel for data collection.
    num_processes = 32

    # Minimum number of observations required for inference.
    min_inference_size = max(1, int(round(num_processes * 0.9)))  # Adjust based on process count.

    # Initialize the PPO learner with the specified environment and configurations.
    learner = Learner(
        build_rocketsim_env,  # Function to create the environment.
        n_proc=num_processes,  # Number of parallel processes.
        min_inference_size=min_inference_size,  # Minimum batch size for inference.
        metrics_logger=metrics_logger,  # Logger for tracking metrics.
        ppo_batch_size=50000,  # Batch size for PPO updates.
        ts_per_iteration=50000,  # Timesteps per training iteration.
        exp_buffer_size=150000,  # Experience buffer size.
        ppo_minibatch_size=50000,  # Minibatch size for PPO updates.
        ppo_ent_coef=0.001,  # Entropy coefficient for exploration.
        ppo_epochs=1,  # Number of PPO epochs per iteration.
        standardize_returns=True,  # Normalize returns for stability.
        standardize_obs=False,  # Disable observation normalization.
        save_every_ts=100_000,  # Save model every 100,000 timesteps.
        timestep_limit=1_000_000_000,  # Maximum timesteps for training.
        log_to_wandb=True  # Enable logging to Weights & Biases.
    )
    learner.learn()