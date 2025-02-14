# Tutorials and Examples

This document provides step-by-step tutorials and examples to help you get started with the `rlgym-ppo` library. Whether you're a beginner or an experienced user, these examples will guide you through various use cases and configurations.

---

## 1. Installation and Setup

Before using the library, ensure you have the required dependencies installed. Follow these steps:

1. Install [RLGym-sim](https://github.com/AechPro/rocket-league-gym-sim).
2. Install PyTorch with CUDA if you plan to use a GPU. Follow the instructions [here](https://pytorch.org/get-started/locally/).
3. Install `rlgym-ppo`:
   ```bash
   pip install git+https://github.com/AechPro/rlgym-ppo
   ```

---

## 2. Creating Custom Environments

The `rlgym-ppo` library allows you to define custom environments tailored to your needs. Below are two examples:

### Example 1: RocketSim Environment

```python
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.action_parsers import ContinuousAction
import numpy as np
from rlgym_sim.utils import common_values

def build_rocketsim_env():
    action_parser = ContinuousAction()
    terminal_conditions = [NoTouchTimeoutCondition(1200), GoalScoredCondition()]

    rewards_to_combine = (VelocityPlayerToBallReward(),
                          VelocityBallToGoalReward(),
                          EventReward(team_goal=1, concede=-1, demo=0.1))
    reward_weights = (0.01, 0.1, 10.0)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    from rlgym_sim import make
    return make(tick_skip=8,
                team_size=1,
                spawn_opponents=True,
                terminal_conditions=terminal_conditions,
                reward_fn=reward_fn,
                obs_builder=obs_builder,
                action_parser=action_parser)
```

### Example 2: RLGym V2 Environment

```python
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine, RLViserRenderer
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym_ppo.util import RLGymV2GymWrapper
import numpy as np

def build_rlgym_v2_env():
    action_parser = RepeatAction(LookupTableAction(), repeats=8)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout=10)

    reward_fn = CombinedReward(
        (GoalReward(), 10),
        (TouchReward(), 0.1)
    )

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / 4096, 1 / 5120, 1 / 2044]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / 2300,
        ang_vel_coef=1 / 5.5
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    )

    return RLGymV2GymWrapper(rlgym_env)
```

---

## 3. Using the Learner Class

The `Learner` class is the core of the `rlgym-ppo` library. It handles training using the PPO algorithm. Here's how to use it:

```python
from rlgym_ppo import Learner

def build_env():
    # Replace with your custom environment function
    return build_rocketsim_env()

learner = Learner(
    env_create_function=build_env,
    n_proc=32,
    min_inference_size=28,
    ppo_batch_size=50000,
    ts_per_iteration=50000,
    exp_buffer_size=150000,
    ppo_minibatch_size=50000,
    ppo_ent_coef=0.001,
    ppo_epochs=1,
    standardize_returns=True,
    standardize_obs=False,
    save_every_ts=100_000,
    timestep_limit=1_000_000_000,
    log_to_wandb=True
)

learner.learn()
```

---

## 4. Best Practices

- **Optimize Hyperparameters**: Experiment with `ppo_batch_size`, `ppo_epochs`, and `learning rates` for better performance.
- **Use Logging**: Enable logging to Weights & Biases (`log_to_wandb=True`) for detailed metrics tracking.
- **Monitor GPU Usage**: Ensure your GPU is utilized efficiently by setting `device="auto"` in the `Learner` class.
- **Debugging**: Use smaller environments and fewer processes (`n_proc`) during debugging to reduce complexity.

---

For more details, refer to the [README](README.md) or the [architecture documentation](architecture.md).
