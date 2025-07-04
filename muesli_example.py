"""
Simple Muesli Example

MUESLI OVERVIEW:
Muesli (Model-Enhanced Stable Policy Optimization) is an advanced RL algorithm that:
- Learns environment dynamics and reward models alongside the policy
- Uses experience reanalysis for ~500% better sample efficiency than PPO
- Employs conservative policy updates for more stable training
- Features target networks for improved learning stability

CONFIGURATION GUIDE:
1. Set USE_OBSERVATION_STACKING = True for temporal understanding (recommended)
2. Adjust STACK_SIZE for memory vs computation trade-off (4-12 frames)
3. Modify network sizes based on your hardware capabilities
4. Tune Muesli hyperparameters for your specific environment
"""

import numpy as np
from rlgym_sim.utils.gamestates import GameState
from action_parser import LookupAction
from rlgym_ppo.util import MetricsLogger


class ObservationStacker:
    """
    Manages temporal observation stacking for improved learning.

    Stacks multiple consecutive observations to provide temporal context,
    enabling the agent to understand motion, velocity, and predict trajectories.

    Benefits:
    - Better ball trajectory prediction
    - Improved opponent movement anticipation
    - Enhanced understanding of game momentum
    """

    def __init__(self, obs_size, stack_size=8):
        self.obs_size = obs_size
        self.stack_size = stack_size
        self.stacked_obs = np.zeros((stack_size, obs_size), dtype=np.float32)

    def reset(self, initial_obs):
        """Reset with initial observation repeated across all frames for smooth start."""
        if initial_obs is not None:
            # Fill stack with initial observation
            for i in range(self.stack_size):
                self.stacked_obs[i] = initial_obs
        else:
            # Clear stack
            self.stacked_obs.fill(0.0)

    def add_observation(self, obs):
        """Add new observation and return flattened stacked observations."""
        self.stacked_obs[1:] = self.stacked_obs[:-1]  # Shift older observations
        self.stacked_obs[0] = obs  # Add newest observation
        return self.stacked_obs.flatten()


class StackedEnvironmentWrapper:
    """
    Environment wrapper that adds observation stacking capability.

    Automatically handles:
    - Multi-agent observation consistency
    - Dynamic observation size detection
    - Proper initialization and reset behavior
    - Memory-efficient observation management

    Usage: Wrap any RLGym environment to add temporal understanding.
    """

    def __init__(self, env_fn, stack_size=8, verbose=True):
        self.env = env_fn()
        self.stack_size = stack_size

        self.action_space = self.env.action_space

        test_obs = self.env.reset()
        if isinstance(test_obs, (list, tuple)) and len(test_obs) > 0:
            obs_size = (
                len(test_obs[0]) if hasattr(test_obs[0], "__len__") else len(test_obs)
            )
        else:
            obs_size = (
                len(test_obs)
                if hasattr(test_obs, "__len__")
                else self._get_default_obs_size()
            )

        self.obs_size = obs_size
        self.stacked_obs_size = obs_size * stack_size
        self.stacker = ObservationStacker(obs_size, stack_size)

        # Create a custom observation space for the stacked observations
        # This is required for compatibility with rlgym-ppo batched agents
        try:
            import gym.spaces as spaces

            self.observation_space = spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.stacked_obs_size,),
                dtype=np.float32,
            )
        except ImportError:
            try:
                import gymnasium.spaces as spaces

                self.observation_space = spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.stacked_obs_size,),
                    dtype=np.float32,
                )
            except ImportError:
                # Create a simple object with shape attribute as fallback
                class SimpleObsSpace:
                    def __init__(self, shape):
                        self.shape = shape

                self.observation_space = SimpleObsSpace((self.stacked_obs_size,))

        if verbose:
            print("Observation Stacking Enabled:")
            print(f"   • Single frame size: {obs_size}")
            print(f"   • Stack size: {stack_size} frames")
            print(f"   • Total observation size: {obs_size * stack_size}")

    def _get_default_obs_size(self):
        """Get default observation size by inspecting the environment's observation space."""
        try:
            if hasattr(self.env, "observation_space") and hasattr(
                self.env.observation_space, "shape"
            ):
                return (
                    self.env.observation_space.shape[0]
                    if len(self.env.observation_space.shape) > 0
                    else self.env.observation_space.shape
                )
            elif hasattr(self.env, "obs_builder") and hasattr(
                self.env.obs_builder, "get_obs_space"
            ):
                return self.env.obs_builder.get_obs_space()
            else:
                test_obs = self.env.reset()
                if hasattr(test_obs, "__len__"):
                    return len(test_obs)
                else:
                    return self._inspect_env_for_obs_size()
        except Exception:
            return self._inspect_env_for_obs_size()

    def _inspect_env_for_obs_size(self):
        """Fallback method to inspect environment for observation size."""
        try:
            if hasattr(self.env, "_obs_builder"):
                obs_builder = self.env._obs_builder
                if hasattr(obs_builder, "get_obs_space"):
                    return obs_builder.get_obs_space()
                elif str(type(obs_builder)).find("DefaultObs") != -1:
                    team_size = getattr(self.env, "_team_size", 1)
                    return 35 + 72 * team_size
            return 89
        except Exception:
            return 89

    def reset(self):
        obs = self.env.reset()
        # Handle multi-agent case - RocketSim returns list of observations
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            single_obs = np.array(obs[0], dtype=np.float32)  # Use first agent
        else:
            single_obs = np.array(obs, dtype=np.float32)

        self.stacker.reset(single_obs)
        stacked_result = self.stacker.stacked_obs.flatten()

        # Return in the same format as the original environment (list for multi-agent)
        if isinstance(obs, (list, tuple)):
            result = []
            for i in range(len(obs)):
                if i == 0:
                    result.append(stacked_result)
                else:
                    original_obs = np.array(obs[i], dtype=np.float32)
                    pseudo_stacked = np.tile(original_obs, self.stack_size)
                    result.append(pseudo_stacked)
            return result
        else:
            return stacked_result

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Handle multi-agent case - RocketSim returns list of observations
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            single_obs = np.array(obs[0], dtype=np.float32)  # Use first agent
        else:
            single_obs = np.array(obs, dtype=np.float32)

        stacked_obs = self.stacker.add_observation(single_obs)

        # Return in the same format as the original environment
        if isinstance(obs, (list, tuple)):
            obs_result = []
            for i in range(len(obs)):
                if i == 0:
                    obs_result.append(stacked_obs)
                else:
                    original_obs = np.array(obs[i], dtype=np.float32)
                    pseudo_stacked = np.tile(original_obs, self.stack_size)
                    obs_result.append(pseudo_stacked)
        else:
            obs_result = stacked_obs

        return obs_result, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class MuesliExampleLogger(MetricsLogger):
    """Simple metrics logger for Muesli training."""

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        metrics = [
            game_state.players[0].car_data.linear_velocity,
            game_state.players[0].car_data.rotation_mtx(),
            game_state.orange_score,
            game_state.ball.linear_velocity,
        ]
        return np.array(metrics, dtype=object)

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        if not collected_metrics:
            return

        avg_player_vel = np.zeros(3)
        avg_ball_vel = np.zeros(3)

        for metric_arrays in collected_metrics:
            player_velocity = np.array(metric_arrays[0])
            avg_player_vel += player_velocity

            if len(metric_arrays) > 3:
                ball_velocity = np.array(metric_arrays[3])
                avg_ball_vel += ball_velocity

        avg_player_vel /= len(collected_metrics)
        avg_ball_vel /= len(collected_metrics)

        report = {
            "player_x_vel": float(avg_player_vel[0]),
            "player_y_vel": float(avg_player_vel[1]),
            "player_z_vel": float(avg_player_vel[2]),
            "player_speed": float(np.linalg.norm(avg_player_vel)),
            "ball_x_vel": float(avg_ball_vel[0]),
            "ball_y_vel": float(avg_ball_vel[1]),
            "ball_z_vel": float(avg_ball_vel[2]),
            "ball_speed": float(np.linalg.norm(avg_ball_vel)),
            "Cumulative Timesteps": cumulative_timesteps,
        }
        wandb_run.log(report)


def build_rocketsim_env():
    """
    Create a basic 1v1 Rocket League training environment.

    Configuration:
    - 1v1 gameplay with opponents
    - 120 Hz physics, 8 tick skip (15 Hz decisions)
    - 10 second timeout per episode
    - Rewards: Ball chase (0.01x), Goal direction (0.1x), Goals/demos (10x)
    - Continuous action space (8 actions: throttle, steer, pitch, yaw, roll, jump, boost, handbrake)

    Returns:
        Configured RLGym environment for Rocket League training.
    """
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import (
        VelocityPlayerToBallReward,
        VelocityBallToGoalReward,
        EventReward,
    )
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        NoTouchTimeoutCondition,
        GoalScoredCondition,
    )
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = ContinuousAction()
    terminal_conditions = [
        NoTouchTimeoutCondition(timeout_ticks),
        GoalScoredCondition(),
    ]

    rewards_to_combine = (
        VelocityPlayerToBallReward(),
        VelocityBallToGoalReward(),
        EventReward(team_goal=1, concede=-1, demo=0.1),
    )
    reward_weights = (0.01, 0.1, 10.0)

    reward_fn = CombinedReward(
        reward_functions=rewards_to_combine, reward_weights=reward_weights
    )

    obs_builder = DefaultObs(
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
    )

    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        action_parser=action_parser,
    )

    return env


def build_stacked_rocketsim_env(stack_size=8):
    """
    RocketSim environment with observation stacking for Muesli.

    Args:
        stack_size (int): Number of consecutive frames to stack for temporal understanding.
                         - 4: Minimal temporal info, faster training
                         - 8: Balanced temporal understanding (recommended)
                         - 12: Rich temporal context, more computation

    Returns:
        Wrapped environment with stacked observations for better temporal learning.
    """
    base_env = build_rocketsim_env()
    stacked_env = StackedEnvironmentWrapper(
        lambda: base_env, stack_size=stack_size, verbose=False
    )
    return stacked_env


# =============================================================================
# ENVIRONMENT CREATION FUNCTIONS - Must be at module level for multiprocessing
# =============================================================================

# Global configuration for environment creation
_STACK_SIZE = 8  # Default value, will be updated by main


def create_stacked_env():
    """Create a stacked RocketSim environment with the configured stack size."""
    return build_stacked_rocketsim_env(stack_size=_STACK_SIZE)


def create_basic_env():
    """Create a basic RocketSim environment without stacking."""
    return build_rocketsim_env()


def set_global_stack_size(stack_size):
    """Set the global stack size for environment creation."""
    global _STACK_SIZE
    _STACK_SIZE = stack_size


if __name__ == "__main__":
    from rlgym_ppo import Learner

    # =============================================================================
    # USER CONFIGURATION - Modify these settings for your needs
    # =============================================================================

    # OBSERVATION STACKING CONFIGURATION
    USE_OBSERVATION_STACKING = True  # Enable for temporal understanding (recommended)
    STACK_SIZE = 8  # Number of frames to stack (4-12 recommended)
    #   - Higher values: Better temporal understanding, more memory/computation
    #   - Lower values: Faster training, less memory usage
    #   - Recommended: 8 for most cases, 4 for limited hardware, 12 for complex scenarios

    # TRAINING CONFIGURATION
    N_PROCESSES = 32  # Number of parallel environments (adjust based on CPU cores)
    #   - More processes: Faster data collection, more CPU usage
    #   - Recommended: 2-4x your CPU cores, but not exceeding 64

    # NETWORK ARCHITECTURE - Modify these layers as needed
    POLICY_LAYERS = (256, 256, 256)  # Policy network layer sizes
    CRITIC_LAYERS = (256, 256, 256)  # Value network layer sizes

    # =============================================================================
    # ENVIRONMENT SETUP
    # =============================================================================

    # Set global configuration for environment creation
    set_global_stack_size(STACK_SIZE)

    # Create environment function based on user configuration
    if USE_OBSERVATION_STACKING:
        test_env = build_stacked_rocketsim_env(stack_size=STACK_SIZE)
        env_function = create_stacked_env
    else:
        test_env = build_rocketsim_env()
        env_function = create_basic_env

    # Detect observation size for info only
    test_obs = test_env.reset()

    if isinstance(test_obs, (list, tuple)):
        if len(test_obs) > 0 and hasattr(test_obs[0], "__len__"):
            actual_obs_size = len(test_obs[0])
        else:
            actual_obs_size = len(test_obs)
    elif hasattr(test_obs, "shape"):
        actual_obs_size = (
            test_obs.shape[-1] if len(test_obs.shape) > 1 else len(test_obs)
        )
    elif hasattr(test_obs, "__len__"):
        actual_obs_size = len(test_obs)
    else:
        actual_obs_size = getattr(test_env, "_get_default_obs_size", lambda: 89)()

    if hasattr(test_env, "stacked_obs_size"):
        actual_obs_size = getattr(test_env, "stacked_obs_size")

    # =============================================================================
    # NETWORK SIZE CONFIGURATION
    # =============================================================================

    print(f"Using network layers: Policy {POLICY_LAYERS}, Critic {CRITIC_LAYERS}")
    print(f"Detected observation size: {actual_obs_size} features")

    # =============================================================================
    # MUESLI TRAINING SETUP
    # =============================================================================

    metrics_logger = MuesliExampleLogger()
    min_inference_size = max(1, int(round(N_PROCESSES * 0.9)))

    learner = Learner(
        env_function,
        n_proc=N_PROCESSES,
        min_inference_size=min_inference_size,
        metrics_logger=metrics_logger,
        # NETWORK ARCHITECTURE
        policy_layer_sizes=POLICY_LAYERS,  # Policy network layers
        critic_layer_sizes=CRITIC_LAYERS,  # Value network layers
        # =============================================================================
        # PPO HYPERPARAMETERS - Core RL training settings
        # =============================================================================
        ppo_batch_size=50000,  # Total samples per training iteration
        #   Larger: More stable gradients, slower training
        #   Smaller: Faster iterations, potentially less stable
        ts_per_iteration=50000,  # Environment steps per iteration
        #   Should match ppo_batch_size for optimal sample usage
        exp_buffer_size=150000,  # Experience replay buffer size
        #   Larger: More diverse experiences, more memory usage
        #   Recommended: 2-4x ppo_batch_size
        ppo_minibatch_size=50000,  # Minibatch size for gradient updates
        #   Larger: More stable gradients, more GPU memory
        #   Smaller: More gradient updates, faster convergence
        ppo_ent_coef=0.001,  # Entropy bonus coefficient (exploration)
        #   Higher: More exploration, less exploitation
        #   Lower: More exploitation, faster convergence
        #   Range: 0.0001 - 0.01
        ppo_epochs=1,  # PPO optimization epochs per iteration
        #   Higher: More thorough optimization, risk of overfitting
        #   Lower: Faster training, potentially suboptimal updates
        # =============================================================================
        # MUESLI-SPECIFIC HYPERPARAMETERS - Advanced model-based features
        # =============================================================================
        use_muesli=True,  # Enable Muesli algorithm
        model_lr=1e-4,  # Learning rate for dynamics/reward models
        #   Lower than policy lr for stability
        #   Range: 1e-5 - 1e-3
        hidden_state_size=256,  # Size of learned state representation
        #   Larger: More expressive models, more computation
        #   Smaller: Faster training, potentially less accurate models
        #   Range: 128 - 512
        n_step_unroll=5,  # Multi-step model learning horizon
        #   Higher: Better long-term predictions, more computation
        #   Lower: Faster training, shorter-term focus
        #   Range: 3 - 10
        target_update_rate=0.005,  # Target network update rate (τ)
        #   Higher: Faster target updates, potentially less stable
        #   Lower: More stable training, slower adaptation
        #   Range: 0.001 - 0.01
        model_loss_weight=0.5,  # Weight for dynamics model loss
        #   Higher: More emphasis on model accuracy
        #   Lower: More emphasis on policy performance
        #   Range: 0.1 - 1.0
        reward_loss_weight=0.5,  # Weight for reward model loss
        #   Higher: More accurate reward predictions
        #   Lower: Faster policy learning
        #   Range: 0.1 - 1.0
        conservative_weight=1.0,  # Conservative policy update strength
        #   Higher: More conservative updates, slower learning
        #   Lower: Faster learning, potentially less stable
        #   Range: 0.5 - 2.0
        reanalysis_ratio=0.5,  # Fraction of reanalyzed experiences
        #   Higher: More sample efficiency, more computation
        #   Lower: Faster training, less sample efficiency
        #   Range: 0.2 - 0.8
        reanalysis_ratio=0.5,  # Fraction of reanalyzed experiences (0.0 to disable)
        replay_buffer_size_muesli=200000, # Size of replay buffer for Muesli reanalysis (if ratio > 0)
        # =============================================================================
        # TRAINING CONTROL SETTINGS
        # =============================================================================
        standardize_returns=True,  # Normalize returns for stable learning
        standardize_obs=False,  # Normalize observations (usually not needed)
        save_every_ts=100_000,  # Save checkpoint frequency
        timestep_limit=1_000_000_000,  # Total training timesteps
        # LOGGING SETTINGS
        log_to_wandb=True,  # Enable Weights & Biases logging
        checkpoints_save_folder="muesli_simple_example_checkpoints",
    )

    print("Starting Muesli Training")
    print(
        f"Configuration: {actual_obs_size} obs features, {len(POLICY_LAYERS)} layer networks"
    )
    if USE_OBSERVATION_STACKING:
        print(f"Observation stacking: {STACK_SIZE} frames for temporal understanding")

    learner.learn()
